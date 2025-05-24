import torch.nn as nn
import torch
from torch.nn import Linear
from utils import *

class EmotionAwareDialogModel(nn.Module):
    def __init__(self, hidden_dim=5):
        super(EmotionAwareDialogModel, self).__init__()
        # self.linear_context = nn.Linear(768, hidden_dim)


    def forward(self, vad_u1, vad_u2, context_embedding, emotion1, emotion2, vad_personality):
        Rc = context_embedding
        E1, E2 = get_emotion_to_VAD(emotion1), get_emotion_to_VAD(emotion2)
        # Avegare Emotion
        E = (E1 + E2) / 2.0

        # Average Attention embeddigs
        A = (vad_u1 + vad_u2) / 2.0

        affective_emb = torch.cat([E, A], dim=1).to(device)

        # Step 3: Personality-Affected Mood Transition
        Pv, Pa, Pd = vad_personality[0], vad_personality[1], vad_personality[2]

        delta_vad = torch.tanh(self.mood_transition(affect_features))
        delta_v, delta_a, delta_d = delta_vad.split(1, dim=-1)

        v_r = init_mood[0] + Pv * delta_v
        a_r = init_mood[1] + Pa * delta_a
        d_r = init_mood[2] + Pd * delta_d

        mood_r = [v_r, a_r, d_r]

        # Step 4: Emotion and Utterance Generation
        mood_proj = torch.tanh(self.linear_mood(mood_r))
        personality_proj = torch.tanh(vad_personality)

        combined = torch.cat([mood_proj, personality_proj, Rc], dim=-1)
        emotion_logits = self.linear_emotion_layer(combined)

        return emotion_logits


def train_model(model, train_loader, num_epoch, learning_rate=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epoch):
        model.train()
