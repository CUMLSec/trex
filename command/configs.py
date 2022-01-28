static_field = 'static'
inst_pos_field = 'inst_pos_emb'  # instruction positional embedding
op_pos_field = 'op_pos_emb'  # opcode/operand positional embedding
arch_field = 'arch_emb'
byte_fields = [f'byte{i}' for i in range(1, 5)]

maskable_fields = [static_field] + byte_fields
aux_fields = [inst_pos_field] + [op_pos_field] + [arch_field]
non_byte_fields = [static_field] + [inst_pos_field] + [op_pos_field] + [arch_field]
fields = non_byte_fields + byte_fields

byte_len = 4
full_attn = False
min_chunk_len = 20
chunk_mask_relax = 0.9
last_layer = -1
cosine_embedding_loss_margin = 0.1
code_value_loss_alpha = 10
