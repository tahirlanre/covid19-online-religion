from transformers import BertPreTrainedModel, BertModel
from transformers.models.bert.modeling_bert import BertOnlyMLMHead, BertPreTrainingHeads

import torch
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn as nn


class DoubleHeadBert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        domain=None,
    ):
        return_dict = False
        outputs = self.bert(input_ids)

        sequence_output, pooled_output = outputs[:2]
        prediction_scores = self.cls(sequence_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        source = domain[0]
        loss = None
        if source == 0:
            if labels is not None:
                if self.num_labels == 1:
                    #  We are doing regression
                    loss_fct = MSELoss()
                    loss = loss_fct(logits.view(-1), labels.view(-1))
                else:
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            output = (logits,)  # + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        else:
            total_loss = None
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                masked_lm_loss = loss_fct(
                    prediction_scores.view(-1, self.config.vocab_size), labels.view(-1)
                )
                total_loss = masked_lm_loss

            output = (prediction_scores,) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output


class DoubleLoss(nn.Module):
    def __init__(self, loss_fn):
        super(DoubleLoss, self).__init__()
        self.loss_fn = loss_fn

    def forward(self, pred, tar, domains):
        if not domains[0]:
            loss = self.loss_fn(pred, tar)
        else:
            loss = torch.tensor(0)
        return loss
