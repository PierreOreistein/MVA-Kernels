import pandas as pd
import numpy as np

from . import EmbeddingDefault
EmbeddingDefault = EmbeddingDefault.EmbeddingDefault


class MotifEmbedding(EmbeddingDefault):
    def __init__(self, motif_lens=[4, 5, 6, 7], mode='rare', do_idf=False):
        super().__init__(embedding=None, hp=None)
        self.motif_lens = motif_lens
        self.mode = mode
        self.do_idf = do_idf

        self.n_documents = None
        self.admissible_motifs = None
        self.motif_to_doc_counts = None
        self.name = "MotifEmbedding"

    def fit(self, X):
        """Fit the data to the Embedding."""

        # Convert X as a dataFrame
        df = pd.DataFrame(np.array(X).reshape((-1, 1)), columns=["seq"])
        n_documents = len(df)

        def _safe_add(dico, key):
            """Add one to the value of key if key is present or just init at 1."""
            if key in dico:
                dico[key] += 1
            else:
                dico[key] = 1

        # Copy of the df
        data_df = df.copy()

        # Initialisation of motif
        motif_to_doc_counts = {}

        # First step: get count of all motifs
        # Loop over all motif length
        for motif_len in self.motif_lens:

            # Loop over all sequence in df
            for sequence in data_df["seq"].values:

                # Extract all sub sequences
                seq_motifs = [sequence[i:i + motif_len] for i in
                              range(len(sequence) - int(motif_len) + 1)]

                # Count the motif saw
                seen_motifs = []
                for motif in seq_motifs:
                    if motif not in seen_motifs:
                        _safe_add(motif_to_doc_counts, motif)
                        seen_motifs.append(motif)

        # Convert the dict of motif_to_doc_counts
        motifs_table = pd.Series(motif_to_doc_counts).to_frame('count')
        motifs_table.loc[:, 'motif'] = motifs_table.index
        motifs_table.loc[:, 'seqlen'] = motifs_table.index.str.len()

        # Extract some motif according to the mode chosen
        if self.mode == 'all':
            admissible_motifs = motifs_table.loc[:, 'motif'].values.squeeze()

        else:
            # Mode: 'rare' or 'frequent'
            # Our objective here is to extract those words

            def _keep_best(my_df, mode=self.mode):
                """Extract only the motif of the given mode."""
                if mode == 'rare':
                    return my_df.loc[my_df['count'] < my_df['count'].mean(), 'motif']
                elif mode == "frequent":
                    return my_df.loc[my_df['count'] > my_df['count'].mean(), 'motif']

            admissible_motifs = (motifs_table
                                 .groupby('seqlen', as_index=False)
                                 .apply(_keep_best).values.squeeze())

        # Update attribute
        self.admissible_motifs = admissible_motifs
        self.n_documents = n_documents
        self.motif_to_doc_counts = motif_to_doc_counts

    def call(self, X, train=False):
        """Fit the data to the Embedding."""

        # If we are during the train phase, fit the data
        if train:
            self.fit(X)

        # Convert the array as a df
        df = pd.DataFrame(np.array(X).reshape((-1, 1)), columns=["seq"])
        data_df = df.copy()

        # Columns to extract after the loop
        columns = []

        for str_seq in self.admissible_motifs:
            # Count
            if self.do_idf:
                idf = np.log(1.0 * self.n_documents / (1.0 + self.motif_to_doc_counts[str_seq]))
                data_df["Embedding_" + str_seq] = data_df["seq"].apply(
                    lambda x: x.count(str_seq) * idf)
            else:
                data_df["Embedding_" + str_seq] = data_df["seq"].apply(lambda x: x.count(str_seq))

            # Append columns
            columns.append("Embedding_" + str_seq)

        # Return data
        return data_df[columns].values
