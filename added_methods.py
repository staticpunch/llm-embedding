class InstructorTrainer(Seq2SeqTrainer):
    def _get_train_sampler(self) :
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        generator = None
        if self.args.world_size <= 1:
            generator = torch.Generator()
            # for backwards compatibility, we generate a seed here (which is sampled from a generator seeded with
            # `args.seed`) if data_seed isn't provided.
            # Further on in this method, we default to `args.seed` instead.
            if self.args.data_seed is None:
                seed = int(torch.empty((), dtype=torch.int64).random_().item())
            else:
                seed = self.args.data_seed
            generator.manual_seed(seed)

        seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed

        if self.args.world_size <= 1:
            return SequentialSampler(self.train_dataset)
        else:
            return DistributedSampler(
                self.train_dataset,
                num_replicas=self.args.world_size,
                rank=self.args.process_index,
                seed=seed,
            )

    def compute_loss(self, model, inputs, return_outputs=False):
        for task_id in inputs['task_id']:
            # assert task_id==inputs['task_id'][0],f"Examples in the same batch should come from the same task, " \
            # f"but task {task_id} and task {inputs['task_id'][0]} are found"
            pass
        cur_results = {}
        for k in ['query', 'pos', 'neg']:
            cur_inputs = {
                'input_ids': inputs[f'{k}_input_ids'],
                'attention_mask': inputs[f'{k}_attention_mask'],
                # 'context_masks': inputs[f'{k}_context_masks'],
            }
            # cur_results[k] = model(cur_inputs)['sentence_embedding']
            # import pdb; pdb.set_trace()
            cur_results[k] = model(**cur_inputs)
        embeddings_query = cur_results['query']
        embeddings_pos = cur_results['pos']
        embeddings_neg = cur_results['neg']

        num = len(embeddings_query)
        all_scores = None
        from torch import nn
        similarity_fct = nn.CosineSimilarity(dim=-1)
        for i in range(0, num):
            anchor_emb = embeddings_query[i].unsqueeze(0)
            pos_emb = embeddings_pos[i].unsqueeze(0)
            cur_score = similarity_fct(anchor_emb, pos_emb) / self.args.cl_temperature

            for j in range(0, num):
                one_neg_emb = embeddings_neg[j].unsqueeze(0)
                one_neg_score = similarity_fct(anchor_emb, one_neg_emb) / self.args.cl_temperature
                cur_score = torch.cat([cur_score, one_neg_score], dim=-1)
            if all_scores is None:
                all_scores = cur_score.unsqueeze(0)
            else:
                all_scores = torch.cat([all_scores, cur_score.unsqueeze(0)], dim=0)

        labels = torch.zeros(all_scores.size(0)).long().to(embeddings_query.device)
        loss = nn.CrossEntropyLoss()(all_scores, labels)

        all_another_scores = None
        for i in range(0, num):
            anchor_emb = embeddings_pos[i].unsqueeze(0)
            pos_emb = embeddings_query[i].unsqueeze(0)
            cur_score = similarity_fct(anchor_emb, pos_emb) / self.args.cl_temperature

            for j in range(0, num):
                if i == j:
                    continue
                one_neg_emb = embeddings_query[j].unsqueeze(0)
                one_neg_score = similarity_fct(anchor_emb, one_neg_emb) / self.args.cl_temperature
                cur_score = torch.cat([cur_score, one_neg_score], dim=-1)
            if all_another_scores is None:
                all_another_scores = cur_score.unsqueeze(0)
            else:
                all_another_scores = torch.cat([all_another_scores, cur_score.unsqueeze(0)], dim=0)
        labels_another = torch.zeros(all_another_scores.size(0)).long().to(embeddings_query.device)
        loss += nn.CrossEntropyLoss()(all_another_scores, labels_another)

        # import pdb; pdb.set_trace()
        if return_outputs:
            return loss, all_scores
        return loss 

@dataclass
class MyTrainingArguments(Seq2SeqTrainingArguments):
    trainable : Optional[str] = field(default="q_proj,v_proj")
    lora_rank : Optional[int] = field(default=8)
    lora_dropout : Optional[float] = field(default=0.1)
    lora_alpha : Optional[float] = field(default=32.)
    modules_to_save : Optional[str] = field(default=None)
    # debug_mode : Optional[bool] = field(default=False)
    peft_path : Optional[str] = field(default=None)
    use_flash_attention_2 : Optional[bool] = field(default=True)
    double_quant: Optional[bool] = field(default=True)
    quant_type: Optional[str] = field(default="nf4")
    load_in_kbits: Optional[int] = field(default=16)
    full_finetuning : Optional[bool] = field(default=False)
