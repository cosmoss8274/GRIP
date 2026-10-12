"""Sequence encoding, receptor mapping and dataset collation."""

import os
import re
from Bio import SeqIO
import pandas as pd
import torch
from torch.utils.data import Dataset

restypes = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "Q",
    "E",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
]
restype_order = {restype: i for i, restype in enumerate(restypes)}
restypes_with_x = restypes + ["X"]
restype_order_with_x = {restype: i for i, restype in enumerate(restypes_with_x)}


def create_hla_mapping(
    hla_fasta_path="data/hla_prot.fasta", mapping_cache_path="data/hla_mapping.csv"
):
    """
    Create mapping between HLA alleles and their sequences from official IPD-IMGT/HLA database
    Args:
        hla_fasta_path: path to hla_prot.fasta from IPD-IMGT/HLA database
        mapping_cache_path: path to save/load the mapping
    """
    if os.path.exists(mapping_cache_path):
        hla_mapping = pd.read_csv(mapping_cache_path, index_col=0).to_dict()["sequence"]
        return hla_mapping
    hla_mapping = {}
    for record in SeqIO.parse(hla_fasta_path, "fasta"):
        header_parts = record.description.split()
        if len(header_parts) < 2:
            continue
        hla_id = header_parts[0]
        allele = header_parts[1]
        if not any((allele.startswith(prefix) for prefix in ["A*", "B*", "C*"])):
            continue
        allele_parts = allele.split(":")
        if len(allele_parts) < 2:
            continue
        main_type = allele_parts[0]
        subtype = allele_parts[1]
        standardized_allele = f"HLA-{main_type}:{subtype}"
        sequence = str(record.seq)
        if not sequence or not all((aa in "ARNDCQEGHILKMFPSTWYV" for aa in sequence)):
            continue
        hla_mapping[standardized_allele] = sequence
        hla_mapping[standardized_allele.replace("HLA-", "")] = sequence
    if len(hla_mapping) == 0:
        raise ValueError("No valid HLA sequences found in FASTA file")
    pd.DataFrame({"sequence": hla_mapping}).to_csv(mapping_cache_path)
    return hla_mapping


class SequenceDataset(Dataset):

    def __init__(
        self,
        list_IDs,
        labels,
        dataFrameIn,
        config,
        integrated_data_path="./data/integrated_tcr_antigen_data.csv",
    ):
        super().__init__()
        c = config.data
        if integrated_data_path is not None:
            self.data = pd.read_csv(integrated_data_path)
        else:
            self.data = dataFrameIn
        self.restype_order_with_x = restype_order_with_x
        self.data = self._preprocess_data(self.data)
        max_length = getattr(c, "max_length", 500)
        self.data = self._filter_by_length(self.data, max_length)
        self.batch_size = c.batch_size
        self.n_channels_emb = c.n_channels_emb
        self.n_channels_seq = c.n_channels_seq
        self.m = c.m
        self.z = c.z
        self.labels = labels
        self.list_IDs = list_IDs
        self.dataFrameIn = dataFrameIn

    def _filter_by_length(self, df, max_length):
        """Filter sequences by maximum length"""
        initial_count = len(df)
        mhc_mask = df["mhc"].apply(
            lambda x: len(x) <= max_length if pd.notna(x) else True
        )
        tcr_mask = df["cdr3_b"].apply(
            lambda x: len(x) <= max_length if pd.notna(x) else True
        )
        epitope_mask = df["epitope_sequence"].apply(
            lambda x: len(x) <= max_length if pd.notna(x) else True
        )
        combined_mask = mhc_mask & tcr_mask & epitope_mask
        filtered_df = df[combined_mask]
        return filtered_df

    def _preprocess_data(self, df):
        """
        Preprocess the data:
        1. Remove rows with empty epitopes
        2. Clean epitope sequences by removing modifications
        """
        df = df[df["epitope_sequence"].notna()]
        df = df[df["epitope_sequence"].str.len() > 0]

        def clean_epitope(seq):
            """Remove modifications from epitope sequence"""
            if pd.isna(seq):
                return seq
            base_seq = seq.split("+")[0].strip()
            base_seq = re.sub("\\([^)]*\\)", "", base_seq)
            valid_aas = set("ARNDCQEGHILKMFPSTWYV")
            base_seq = "".join((aa for aa in base_seq if aa in valid_aas))
            return base_seq

        df["epitope_sequence"] = df["epitope_sequence"].apply(clean_epitope)
        df = df[df["epitope_sequence"].str.len() > 0]
        return df

    def __len__(self):
        return len(self.data)

    def encode_sequence(self, seq, as_target=False):
        """
        Encode an amino acid sequence into one-hot format or target indices.

        Args:
            seq: Amino acid sequence string
            as_target: If True, return indices instead of one-hot encoding

        Returns:
            torch.Tensor: One-hot encoded sequence [seq_len, n_channels_seq] or
                         target indices [seq_len] if as_target=True
        """
        if pd.isna(seq) or len(seq) == 0:
            return (
                torch.zeros((0, self.n_channels_seq), dtype=torch.float32)
                if not as_target
                else torch.zeros(0, dtype=torch.long)
            )
        if as_target:
            return torch.tensor(
                [
                    self.restype_order_with_x.get(
                        aa, len(self.restype_order_with_x) - 1
                    )
                    for aa in seq
                ],
                dtype=torch.long,
            )
        else:
            encoded = torch.zeros(len(seq), self.n_channels_seq, dtype=torch.float32)
            for index, aa in enumerate(seq):
                if aa in self.restype_order_with_x:
                    aa_index = self.restype_order_with_x[aa]
                    encoded[index, aa_index] = 1
                else:
                    encoded[index, -1] = 1
            return encoded

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        epitope_seq = row["epitope_sequence"]
        epitope_len = len(epitope_seq)
        mhc_seq = row["mhc"] if pd.notna(row["mhc"]) else ""
        l_v = row["l_v"]
        l_j = row["l_j"]
        h_v = row["h_v"]
        cdr3_b = row["cdr3_b"]
        h_j = row["h_j"]
        mhc_encoded = self.encode_sequence(mhc_seq)
        l_v_encoded = (
            self.encode_sequence(l_v)
            if not pd.isna(l_v)
            else torch.zeros(0, self.n_channels_seq)
        )
        l_j_encoded = (
            self.encode_sequence(l_j)
            if not pd.isna(l_j)
            else torch.zeros(0, self.n_channels_seq)
        )
        h_v_encoded = (
            self.encode_sequence(h_v)
            if not pd.isna(h_v)
            else torch.zeros(0, self.n_channels_seq)
        )
        cdr3_b_encoded = (
            self.encode_sequence(cdr3_b)
            if not pd.isna(cdr3_b)
            else torch.zeros(0, self.n_channels_seq)
        )
        h_j_encoded = (
            self.encode_sequence(h_j)
            if not pd.isna(h_j)
            else torch.zeros(0, self.n_channels_seq)
        )
        mhc_len = len(mhc_seq) if not pd.isna(mhc_seq) else 0
        l_v_len = len(l_v) if not pd.isna(l_v) else 0
        l_j_len = len(l_j) if not pd.isna(l_j) else 0
        h_v_len = len(h_v) if not pd.isna(h_v) else 0
        cdr3_b_len = len(cdr3_b) if not pd.isna(cdr3_b) else 0
        h_j_len = len(h_j) if not pd.isna(h_j) else 0
        item = {
            "mhc_encoded": mhc_encoded,
            "l_v_encoded": l_v_encoded,
            "l_j_encoded": l_j_encoded,
            "h_v_encoded": h_v_encoded,
            "cdr3_b_encoded": cdr3_b_encoded,
            "h_j_encoded": h_j_encoded,
            "mhc_ri": torch.arange(mhc_len) if mhc_len > 0 else torch.zeros(0),
            "l_v_ri": torch.arange(l_v_len) if l_v_len > 0 else torch.zeros(0),
            "l_j_ri": torch.arange(l_j_len) if l_j_len > 0 else torch.zeros(0),
            "h_v_ri": torch.arange(h_v_len) if h_v_len > 0 else torch.zeros(0),
            "cdr3_b_ri": torch.arange(cdr3_b_len) if cdr3_b_len > 0 else torch.zeros(0),
            "h_j_ri": torch.arange(h_j_len) if h_j_len > 0 else torch.zeros(0),
            "epitope_ri": torch.arange(epitope_len),
            "mhc_types": torch.zeros(mhc_len) if mhc_len > 0 else torch.zeros(0),
            "l_v_types": torch.full((l_v_len,), 1) if l_v_len > 0 else torch.zeros(0),
            "l_j_types": torch.full((l_j_len,), 2) if l_j_len > 0 else torch.zeros(0),
            "h_v_types": torch.full((h_v_len,), 3) if h_v_len > 0 else torch.zeros(0),
            "cdr3_b_types": (
                torch.full((cdr3_b_len,), 4) if cdr3_b_len > 0 else torch.zeros(0)
            ),
            "h_j_types": torch.full((h_j_len,), 5) if h_j_len > 0 else torch.zeros(0),
            "epitope_types": torch.full((epitope_len,), 6),
            "mhc_len": mhc_len,
            "l_v_len": l_v_len,
            "l_j_len": l_j_len,
            "h_v_len": h_v_len,
            "cdr3_b_len": cdr3_b_len,
            "h_j_len": h_j_len,
            "epitope_len": epitope_len,
        }
        target = self.encode_sequence(epitope_seq, as_target=True)
        return (item, target)

    def collate_fn(self, batch):
        """Collate function handling all TCR features"""
        batch_size = len(batch)
        max_mhc_len = max((x[0]["mhc_len"] for x in batch))
        max_epitope_len = max((x[0]["epitope_len"] for x in batch))
        max_l_v_len = max((x[0]["l_v_len"] for x in batch))
        max_l_j_len = max((x[0]["l_j_len"] for x in batch))
        max_h_v_len = max((x[0]["h_v_len"] for x in batch))
        max_cdr3_b_len = max((x[0]["cdr3_b_len"] for x in batch))
        max_h_j_len = max((x[0]["h_j_len"] for x in batch))
        mhc_encoded = (
            torch.zeros(batch_size, max_mhc_len, self.n_channels_seq)
            if max_mhc_len > 0
            else torch.zeros(batch_size, 0, self.n_channels_seq)
        )
        l_v_encoded = (
            torch.zeros(batch_size, max_l_v_len, self.n_channels_seq)
            if max_l_v_len > 0
            else torch.zeros(batch_size, 0, self.n_channels_seq)
        )
        l_j_encoded = (
            torch.zeros(batch_size, max_l_j_len, self.n_channels_seq)
            if max_l_j_len > 0
            else torch.zeros(batch_size, 0, self.n_channels_seq)
        )
        h_v_encoded = (
            torch.zeros(batch_size, max_h_v_len, self.n_channels_seq)
            if max_h_v_len > 0
            else torch.zeros(batch_size, 0, self.n_channels_seq)
        )
        cdr3_b_encoded = (
            torch.zeros(batch_size, max_cdr3_b_len, self.n_channels_seq)
            if max_cdr3_b_len > 0
            else torch.zeros(batch_size, 0, self.n_channels_seq)
        )
        h_j_encoded = (
            torch.zeros(batch_size, max_h_j_len, self.n_channels_seq)
            if max_h_j_len > 0
            else torch.zeros(batch_size, 0, self.n_channels_seq)
        )
        mhc_ri = (
            torch.zeros(batch_size, max_mhc_len)
            if max_mhc_len > 0
            else torch.zeros(batch_size, 0)
        )
        epitope_ri = torch.zeros(batch_size, max_epitope_len)
        l_v_ri = (
            torch.zeros(batch_size, max_l_v_len)
            if max_l_v_len > 0
            else torch.zeros(batch_size, 0)
        )
        l_j_ri = (
            torch.zeros(batch_size, max_l_j_len)
            if max_l_j_len > 0
            else torch.zeros(batch_size, 0)
        )
        h_v_ri = (
            torch.zeros(batch_size, max_h_v_len)
            if max_h_v_len > 0
            else torch.zeros(batch_size, 0)
        )
        cdr3_b_ri = (
            torch.zeros(batch_size, max_cdr3_b_len)
            if max_cdr3_b_len > 0
            else torch.zeros(batch_size, 0)
        )
        h_j_ri = (
            torch.zeros(batch_size, max_h_j_len)
            if max_h_j_len > 0
            else torch.zeros(batch_size, 0)
        )
        mhc_types = (
            torch.zeros(batch_size, max_mhc_len)
            if max_mhc_len > 0
            else torch.zeros(batch_size, 0)
        )
        epitope_types = torch.zeros(batch_size, max_epitope_len)
        l_v_types = (
            torch.zeros(batch_size, max_l_v_len)
            if max_l_v_len > 0
            else torch.zeros(batch_size, 0)
        )
        l_j_types = (
            torch.zeros(batch_size, max_l_j_len)
            if max_l_j_len > 0
            else torch.zeros(batch_size, 0)
        )
        h_v_types = (
            torch.zeros(batch_size, max_h_v_len)
            if max_h_v_len > 0
            else torch.zeros(batch_size, 0)
        )
        cdr3_b_types = (
            torch.zeros(batch_size, max_cdr3_b_len)
            if max_cdr3_b_len > 0
            else torch.zeros(batch_size, 0)
        )
        h_j_types = (
            torch.zeros(batch_size, max_h_j_len)
            if max_h_j_len > 0
            else torch.zeros(batch_size, 0)
        )
        max_target_len = max((len(t) for _, t in batch))
        target_tensor = torch.zeros(batch_size, max_target_len, dtype=torch.long)
        for i, (item, target) in enumerate(batch):
            target_len = len(target)
            target_tensor[i, :target_len] = target
            if item["mhc_len"] > 0:
                mhc_encoded[i, : item["mhc_len"]] = item["mhc_encoded"]
                mhc_ri[i, : item["mhc_len"]] = item["mhc_ri"]
                mhc_types[i, : item["mhc_len"]] = item["mhc_types"]
            if item["epitope_len"] > 0:
                epitope_ri[i, : item["epitope_len"]] = item["epitope_ri"]
                epitope_types[i, : item["epitope_len"]] = item["epitope_types"]
            if item["l_v_len"] > 0:
                l_v_encoded[i, : item["l_v_len"]] = item["l_v_encoded"]
                l_v_ri[i, : item["l_v_len"]] = item["l_v_ri"]
                l_v_types[i, : item["l_v_len"]] = item["l_v_types"]
            if item["l_j_len"] > 0:
                l_j_encoded[i, : item["l_j_len"]] = item["l_j_encoded"]
                l_j_ri[i, : item["l_j_len"]] = item["l_j_ri"]
                l_j_types[i, : item["l_j_len"]] = item["l_j_types"]
            if item["h_v_len"] > 0:
                h_v_encoded[i, : item["h_v_len"]] = item["h_v_encoded"]
                h_v_ri[i, : item["h_v_len"]] = item["h_v_ri"]
                h_v_types[i, : item["h_v_len"]] = item["h_v_types"]
            if item["cdr3_b_len"] > 0:
                cdr3_b_encoded[i, : item["cdr3_b_len"]] = item["cdr3_b_encoded"]
                cdr3_b_ri[i, : item["cdr3_b_len"]] = item["cdr3_b_ri"]
                cdr3_b_types[i, : item["cdr3_b_len"]] = item["cdr3_b_types"]
            if item["h_j_len"] > 0:
                h_j_encoded[i, : item["h_j_len"]] = item["h_j_encoded"]
                h_j_ri[i, : item["h_j_len"]] = item["h_j_ri"]
                h_j_types[i, : item["h_j_len"]] = item["h_j_types"]
        return (
            {
                "mhc_encoded": mhc_encoded,
                "l_v_encoded": l_v_encoded,
                "l_j_encoded": l_j_encoded,
                "h_v_encoded": h_v_encoded,
                "cdr3_b_encoded": cdr3_b_encoded,
                "h_j_encoded": h_j_encoded,
                "mhc_ri": mhc_ri,
                "epitope_ri": epitope_ri,
                "l_v_ri": l_v_ri,
                "l_j_ri": l_j_ri,
                "h_v_ri": h_v_ri,
                "cdr3_b_ri": cdr3_b_ri,
                "h_j_ri": h_j_ri,
                "mhc_types": mhc_types,
                "epitope_types": epitope_types,
                "l_v_types": l_v_types,
                "l_j_types": l_j_types,
                "h_v_types": h_v_types,
                "cdr3_b_types": cdr3_b_types,
                "h_j_types": h_j_types,
                "mhc_lengths": torch.tensor([x[0]["mhc_len"] for x in batch]),
                "epitope_lengths": torch.tensor([x[0]["epitope_len"] for x in batch]),
                "l_v_lengths": torch.tensor([x[0]["l_v_len"] for x in batch]),
                "l_j_lengths": torch.tensor([x[0]["l_j_len"] for x in batch]),
                "h_v_lengths": torch.tensor([x[0]["h_v_len"] for x in batch]),
                "cdr3_b_lengths": torch.tensor([x[0]["cdr3_b_len"] for x in batch]),
                "h_j_lengths": torch.tensor([x[0]["h_j_len"] for x in batch]),
                "aatype": target_tensor,
            },
            target_tensor,
        )
