class FPN(nn.Module):
    def __init__(self, args, d_model=256, heads=4):
        super(FPN, self).__init__()
        self.fp_type = getattr(args, 'fp_type', 'mixed')
        self.fp_changebit = getattr(args, 'fp_changebit', None)
        self.cuda = args.cuda
        self.dropout_rate = args.dropout
        self.hidden_dim = args.hidden_size

        # 各指纹维度
        self.dim_maccs = 167
        self.dim_erg = 441
        self.dim_pub = 881

        # 映射层，投影到统一d_model
        self.project_maccs = nn.Linear(self.dim_maccs, d_model)
        self.project_erg = nn.Linear(self.dim_erg, d_model)
        self.project_pub = nn.Linear(self.dim_pub, d_model)

        # MultiheadAttention，不带batch_first（旧版兼容）
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=heads)

        self.norm1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(self.dropout_rate)

        self.fc_out = nn.Sequential(
            nn.Linear(d_model, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Dropout(self.dropout_rate)
        )

    def forward(self, smiles):
        batch_fp_maccs = []
        batch_fp_erg = []
        batch_fp_pub = []

        for smi in smiles:
            mol = Chem.MolFromSmiles(smi)
            fp_maccs = AllChem.GetMACCSKeysFingerprint(mol)
            fp_erg = AllChem.GetErGFingerprint(mol, fuzzIncrement=0.3, maxPath=21, minPath=1)
            fp_pub = GetPubChemFPs(mol)

            batch_fp_maccs.append(np.array(fp_maccs))
            batch_fp_erg.append(np.array(fp_erg))
            batch_fp_pub.append(np.array(fp_pub))

        batch_fp_maccs = torch.tensor(np.array(batch_fp_maccs), dtype=torch.float32)
        batch_fp_erg = torch.tensor(np.array(batch_fp_erg), dtype=torch.float32)
        batch_fp_pub = torch.tensor(np.array(batch_fp_pub), dtype=torch.float32)

        if self.fp_changebit is not None and self.fp_changebit != 0:
            batch_fp_maccs[:, self.fp_changebit - 1] = 1.0

        if self.cuda:
            batch_fp_maccs = batch_fp_maccs.cuda()
            batch_fp_erg = batch_fp_erg.cuda()
            batch_fp_pub = batch_fp_pub.cuda()

        proj_maccs = self.project_maccs(batch_fp_maccs)
        proj_erg = self.project_erg(batch_fp_erg)
        proj_pub = self.project_pub(batch_fp_pub)

        # 拼接序列 (B, 3, d_model) -> 转换成 (3, B, d_model) 以兼容旧版MultiheadAttention
        seq = torch.stack([proj_maccs, proj_erg, proj_pub], dim=1).transpose(0, 1)

        attn_out, _ = self.self_attn(seq, seq, seq)
        seq = seq + self.dropout(attn_out)
        seq = self.norm1(seq)

        # 转回 (B, 3, d_model)
        seq = seq.transpose(0, 1)

        # 池化
        pooled = seq.mean(dim=1)

        output = self.fc_out(pooled)
        return output


    def forward(self, smiles):
        batch_fp_maccs = []
        batch_fp_erg = []
        batch_fp_pub = []

        for smi in smiles:
            mol = Chem.MolFromSmiles(smi)

            fp_maccs = AllChem.GetMACCSKeysFingerprint(mol)
            fp_erg = AllChem.GetErGFingerprint(mol, fuzzIncrement=0.3, maxPath=21, minPath=1)
            fp_pub = GetPubChemFPs(mol)

            arr_maccs = np.array(fp_maccs)
            arr_erg = np.array(fp_erg)
            arr_pub = np.array(fp_pub)

            batch_fp_maccs.append(arr_maccs)
            batch_fp_erg.append(arr_erg)
            batch_fp_pub.append(arr_pub)

        batch_fp_maccs = torch.tensor(np.array(batch_fp_maccs), dtype=torch.float32)
        batch_fp_erg = torch.tensor(np.array(batch_fp_erg), dtype=torch.float32)
        batch_fp_pub = torch.tensor(np.array(batch_fp_pub), dtype=torch.float32)

        if self.fp_changebit is not None and self.fp_changebit != 0:
            batch_fp_maccs[:, self.fp_changebit - 1] = 1.0

        if self.cuda:
            batch_fp_maccs = batch_fp_maccs.cuda()
            batch_fp_erg = batch_fp_erg.cuda()
            batch_fp_pub = batch_fp_pub.cuda()

        # 投影到统一维度
        proj_maccs = self.project_maccs(batch_fp_maccs)
        proj_erg = self.project_erg(batch_fp_erg)
        proj_pub = self.project_pub(batch_fp_pub)

        # 堆叠成序列 (B, 3, d_model)
        seq = torch.stack([proj_maccs, proj_erg, proj_pub], dim=1)

        # 自注意力
        attn_out, _ = self.self_attn(seq, seq, seq)
        seq = self.norm1(seq + self.dropout(attn_out))

        # 池化得到整体表示 (B, d_model)
        pooled = seq.mean(dim=1)

        # 通过两层线性映射输出最终指纹特征
        output = self.fc_out(pooled)
        return output