import torch
from transformers import AutoModelForTokenClassification


class JointIntentAndSlotModel(torch.nn.Module):
    def __init__(self, num_intents, num_slots, base_model="distilbert-base-uncased"):
        """``base_model`` is a hub name or a local dir of a pretrained encoder
        (training starts from it: DistilBERT, or for the question model an
        Ettin / ModernBERT encoder, q1 round 4) or a fine-tuned checkpoint dir
        (inference loads the encoder + slot head directly, with no network
        access). Its token classifier is built by AutoModelForTokenClassification:
        a DistilBERT checkpoint gives DistilBertForTokenClassification, as before."""
        super().__init__()
        self.bert_for_slots = AutoModelForTokenClassification.from_pretrained(
            base_model, num_labels=num_slots
        )
        self.intent_classifier = torch.nn.Linear(
            self.bert_for_slots.config.hidden_size, num_intents
        )

    @property
    def encoder(self):
        """The bare encoder (``.distilbert`` for DistilBERT, ``.model`` for ModernBERT)."""
        return self.bert_for_slots.base_model

    def forward(self, input_ids, attention_mask, intent_labels=None, slot_labels=None):
        bert_output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_output.last_hidden_state

        cls_token_output = sequence_output[:, 0, :]
        intent_logits = self.intent_classifier(cls_token_output)

        slot_logits = self.bert_for_slots.classifier(sequence_output)

        total_loss = 0
        if intent_labels is not None and slot_labels is not None:
            intent_loss = torch.nn.CrossEntropyLoss()(intent_logits, intent_labels.view(-1))
            slot_loss = torch.nn.CrossEntropyLoss()(
                slot_logits.view(-1, self.bert_for_slots.num_labels), slot_labels.view(-1)
            )
            total_loss = intent_loss + slot_loss

        return total_loss, intent_logits, slot_logits
