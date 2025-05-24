import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from utils import get_emotion_to_VAD

class DialogueDataset(Dataset):
    def __init__(self, tsv_path, embedding_path_u1_u2, embedding_path_u1_u2_u3):
        # Load the TSV
        self.df = pd.read_csv(tsv_path, sep='\t')

        # Load context embeddings
        self.context_embeddings_u1_u2 = torch.tensor(np.load(embedding_path_u1_u2), dtype=torch.float32)
        self.context_embeddings_u1_u2_u3 = torch.tensor(np.load(embedding_path_u1_u2_u3), dtype=torch.float32)
        self.utterance1 = self.df["Utterance_1"].tolist()
        self.utterance2 = self.df["Utterance_2"].tolist()
        # Extract relevant fields
        self.emotion1 = self.df["Emotion_1"].tolist()
        self.emotion2 = self.df["Emotion_2"].tolist()
        self.responses = self.df["Utterance_3"].tolist()
        self.target_emotions = self.df["Emotion_3"].tolist()
        self.personalities = torch.tensor(
            self.df["Personality"].apply(eval).tolist(), dtype=torch.float32
        )
        self.personalities_vad = self.personality_to_vad()

        # Precompute VAD embeddings for utterances add_vad_utterances_with_multihead_attention
        self.vad_utterance1, self.vad_utterance2, self.vad_utterance3 = self.add_vad_utterances_with_attention()

        # Pre compute VAD for emotions
        self.vad_emotion1, self.vad_emotion2, self.vad_emotion3 = self.add_vad_emotions()

    def __len__(self):
        return len(self.df)

    def personality_to_vad(self):
        O = self.personalities[:, 0]
        C = self.personalities[:, 1]
        E = self.personalities[:, 2]
        A = self.personalities[:, 3]
        N = self.personalities[:, 4]

        valence = 0.21 * E + 0.59 * A + 0.19 * N
        arousal = 0.15 * O + 0.30 * A - 0.57 * N
        dominance = 0.25 * O + 0.17 * C + 0.60 * E - 0.32 * A

        return torch.cat(
            (valence.unsqueeze(-1), arousal.unsqueeze(-1), dominance.unsqueeze(-1)), dim=1
        )

    def compute_emotion_aware_embedding(self,utterance, emotion_label, vad_lexicon):
        import pandas as pd
        import numpy as np
        from nltk.tokenize import word_tokenize
        from collections import defaultdict
        from utils import EMOTION_TO_VAD, load_nrc_vad
        tokens = word_tokenize(utterance.lower())
        token_vads = []

        for token in tokens:
            if token in vad_lexicon:
                token_vads.append(vad_lexicon[token])
            else:
                token_vads.append(np.zeros(3))  # unknown word = [0,0,0]

        token_vads = np.stack(token_vads)  # Shape: [m, 3]
        ei = EMOTION_TO_VAD.get(emotion_label.lower(), np.array([0.5, 0.5, 0.5]))  # [3]

        # Attention weights: softmax of dot product between each token-VAD and emotion-VAD
        logits = np.dot(token_vads, ei)  # shape: [m]
        attention_weights = np.exp(logits) / np.sum(np.exp(logits))  # softmax

        # Emotion-aware sentence embedding: weighted sum
        sentence_embedding = np.sum(attention_weights[:, np.newaxis] * token_vads, axis=0)  # [3]

        return sentence_embedding

    def compute_emotion_aware_embedding_multihead(self, utterance, emotion_label, vad_lexicon):
        import numpy as np
        from nltk.tokenize import word_tokenize
        from utils import EMOTION_TO_VAD

        tokens = word_tokenize(utterance.lower())
        token_vads = []

        # Step 1: Get token-level VAD vectors
        for token in tokens:
            if token in vad_lexicon:
                token_vads.append(vad_lexicon[token])  # shape: [3]
            else:
                token_vads.append(np.zeros(3))

        token_vads = np.stack(token_vads)  # Shape: [m, 3]

        # Step 2: Get emotion-level VAD vector
        ei = EMOTION_TO_VAD.get(emotion_label.lower(), np.array([0.5, 0.5, 0.5]))  # shape: [3]
        val_ei, aro_ei, dom_ei = ei[0], ei[1], ei[2]

        # Step 3: Multi-head attention (V, A, D as separate heads)
        val_logits = token_vads[:, 0] * val_ei
        aro_logits = token_vads[:, 1] * aro_ei
        dom_logits = token_vads[:, 2] * dom_ei

        def softmax(x):
            exps = np.exp(x - np.max(x))
            return exps / np.sum(exps)

        val_attn = softmax(val_logits)
        aro_attn = softmax(aro_logits)
        dom_attn = softmax(dom_logits)

        # Step 4: Weighted sum (each dimension separately)
        val_output = np.sum(val_attn * token_vads[:, 0])
        aro_output = np.sum(aro_attn * token_vads[:, 1])
        dom_output = np.sum(dom_attn * token_vads[:, 2])

        # Step 5: Final embedding
        sentence_embedding = np.array([val_output, aro_output, dom_output])  # shape: [3]

        return sentence_embedding

    def add_vad_utterances_with_multihead_attention(self):
        from utils import EMOTION_TO_VAD, load_nrc_vad
        vad_lexicon = load_nrc_vad("NRC-VAD-Lexicon-v2.1.txt")

        # 3. Compute token VADs and attention

        vad_u1 = []
        vad_u2 = []
        vad_u3 = []
        for i in range(len(self.df)):
            vad1 = self.compute_emotion_aware_embedding_multihead(self.utterance1[i], self.emotion1[i], vad_lexicon)
            vad2 = self.compute_emotion_aware_embedding_multihead(self.utterance2[i], self.emotion2[i], vad_lexicon)
            vad3 = self.compute_emotion_aware_embedding_multihead(self.responses[i], self.target_emotions[i], vad_lexicon)
            vad_u1.append(torch.tensor(vad1, dtype=torch.float32))
            vad_u2.append(torch.tensor(vad2, dtype=torch.float32))
            vad_u3.append(torch.tensor(vad3, dtype=torch.float32))
        return vad_u1, vad_u2, vad_u3

    def add_vad_emotions(self):
        vad_e1 = []
        vad_e2 = []
        vad_e3 = []
        for i in range(len(self.df)):

            vad1 = get_emotion_to_VAD(self.emotion1[i])
            vad2 = get_emotion_to_VAD(self.emotion2[i])
            vad3 = get_emotion_to_VAD(self.target_emotions[i])
            vad_e1.append(torch.tensor(vad1, dtype=torch.float32))
            vad_e2.append(torch.tensor(vad2, dtype=torch.float32))
            vad_e3.append(torch.tensor(vad3, dtype=torch.float32))
        return vad_e1, vad_e2, vad_e3

    def add_vad_utterances_with_attention(self):
        from utils import EMOTION_TO_VAD, load_nrc_vad
        vad_lexicon = load_nrc_vad("NRC-VAD-Lexicon-v2.1.txt")

        # 3. Compute token VADs and attention

        vad_u1 = []
        vad_u2 = []
        vad_u3 = []
        for i in range(len(self.df)):
            vad1 = self.compute_emotion_aware_embedding(self.utterance1[i], self.emotion1[i], vad_lexicon)
            vad2 = self.compute_emotion_aware_embedding(self.utterance2[i], self.emotion2[i], vad_lexicon)
            vad3 = self.compute_emotion_aware_embedding(self.responses[i], self.target_emotions[i], vad_lexicon)
            vad_u1.append(torch.tensor(vad1, dtype=torch.float32))
            vad_u2.append(torch.tensor(vad2, dtype=torch.float32))
            vad_u3.append(torch.tensor(vad3, dtype=torch.float32))
        return vad_u1, vad_u2, vad_u3



    def __getitem__(self, idx):
        return {
            "utterance_1": self.utterance1[idx],
            "utterance_2": self.utterance2[idx],
            "utterance_3": self.responses[idx],
            "emotion_1": self.emotion1[idx],
            "emotion_2": self.emotion2[idx],
            "response": self.responses[idx],
            "target_emotion": self.target_emotions[idx],
            "personality": torch.tensor(self.personalities[idx], dtype=torch.float32),
            "vad_personality": torch.tensor(self.personalities_vad[idx], dtype=torch.float32),
            "vad_u1": self.vad_utterance1[idx],
            "vad_u2": self.vad_utterance2[idx],
            "vad_u3": self.vad_utterance3[idx],
            "vad_e1": self.vad_emotion1[idx],
            "vad_e2": self.vad_emotion2[idx],
            "vad_e3": self.vad_emotion3[idx],
            "context_embeddings": self.context_embeddings_u1_u2[idx],
            "context_embeddings_u3": self.context_embeddings_u1_u2_u3[idx],
        }
