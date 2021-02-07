
# 参数微调
model = None
bert_learning_rate = 2e-5
learning_rate = 1e-3
bert_parameters = model.bert.named_parameters()
start_parameters = model.start_fc.named_parameters()
end_parameters = model.end_fc.named_parameters()

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in bert_parameters if not any(nd in n for nd in no_decay)],
    'weight_decay': 0.01, 'lr':bert_learning_rate},
    {'params': [p for n, p in bert_parameters if any(nd in n for nd in no_decay)],
    'weight_decay': 0.0, 'lr':bert_learning_rate},

    {'params': [p for n, p in start_parameters if not any(nd in n for nd in no_decay)],
        'weight_decay': 0.01, 'lr':learning_rate},
    {'params': [p for n, p in start_parameters if any(nd in n for nd in no_decay)],
        'weight_decay': 0.0, 'lr':learning_rate},

    {'params': [p for n, p in end_parameters if not any(nd in n for nd in no_decay)],
        'weight_decay': 0.01, 'lr':learning_rate},
    {'params': [p for n, p in end_parameters if any(nd in n for nd in no_decay)],
        'weight_decay': 0.0, 'lr':learning_rate},
]
