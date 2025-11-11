import pandas as pd

class KMeans:
    """
    Custom KMeans-like class.
    Instead of centroid-based clustering, this uses rule-based segmentation.
    """

    def __init__(self, n_clusters=None):
        self.n_clusters = n_clusters  # Not used in rules, just to mimic API
        self.labels_ = None
        self.cluster_names_ = None

    def fit(self, df: pd.DataFrame, k_type: str = 'rough'):
        """
        Apply rule-based segmentation on DataFrame.
        """

        # --- Step 1: Segment Functions ---
        def segment_01(row):  # High Networth
            return (
                row.get('total_balance_flag', '') == 'high' and
                row.get('total_products', 0) >= 1
            )

        def segment_02(row):  # High Engagement Credit Customer
            return (
                row.get('credit_card_total_transaction_count_flag') != 'None'
                and row.get('total_credit_cards', 0) >= 1
                and row.get('credit_card_total_transaction_amount_flag', '') == 'high'
            )

        def segment_03(row):  # High Frequency Debit Customer
            return (
                row.get('debit_card_total_transaction_count_flag', '') == 'high'
                and row.get('dc_is_debit_card_holder', '').lower() == 'yes'
            )

        def segment_04(row):  # Financially Diverse Consumer
            return (
                row.get('total_products', 0) >= 2 and
                (
                    row.get('debit_card_total_transaction_count_flag', '') in ['medium-high', 'medium-low'] or
                    row.get('credit_card_total_transaction_count_flag', '') in ['medium-high', 'medium-low'] or
                    row.get('total_transaction_count_flag', '') in ['high', 'medium-high', 'medium-low']
                ) and
                (
                    row.get('debit_card_total_transaction_amount_flag', '') in ['medium-high', 'medium-low'] or
                    row.get('credit_card_total_transaction_amount_flag', '') in ['medium-high', 'medium-low']
                ) and
                (
                    row.get('dc_pos_tran_cnt', 0) > 0 or
                    row.get('dc_atmwith_cnt', 0) > 0 or
                    row.get('dc_online_tran_cnt', 0) > 0 or
                    row.get('cc_online_txn_cnt_total', 0) > 0 or
                    row.get('cc_pos_txn_cnt_total', 0) > 0
                )
            )

        def segment_05(row):  # Loyal High Value Customer
            return (
                (
                    row.get('total_balance_flag', '') != 'None'
                    and (
                        row.get('credit_card_total_transaction_amount_flag', '') in ['medium-high', 'high']
                        or row.get('debit_card_total_transaction_amount_flag', '') in ['medium-high', 'high']
                    )
                )
                and row.get('urb_customer_tenure_months', 0) > 12
            )

        def segment_06(row):  # Everyday Online Transactor
            return (
                row.get('debit_card_total_transaction_count_flag', '') != 'None'
                and (
                    (
                        row.get('cc_online_txn_cnt_total_flag', '') not in ['None', 'low']
                    )
                    or (
                        row.get('wal_total_txns_total_flag', '') not in ['None', 'low']
                    )
                    or (
                        row.get('dc_online_tran_cnt_total_flag', '') not in ['None', 'low']
                    )
                )
            )

        def segment_07(row):  # Conservative Financial Traditionalist
            return (
                (
                    row.get('is_low_transaction_customer', False) or row.get('is_conservative_balance_customer', False)
                )
                and row.get('total_products', 0) >= 1
                or (
                    row.get('prd_classic_debit_card', 0) == 1
                    or row.get('prd_hassala', 0) == 1
                    or row.get('prd_alkanz', 0) == 1
                )
            )

        # --- Step 2: Cluster Assignment ---
        def assign_segment(row):
            if segment_01(row):
                return 0
            elif segment_02(row):
                return 1
            elif segment_03(row):
                return 2
            elif segment_05(row):
                return 3
            elif segment_04(row):
                return 4
            elif segment_06(row):
                return 5
            else:
                return 6

        # --- Step 3: Apply on DataFrame ---
        df['Cluster'] = df.apply(lambda row: assign_segment(row), axis=1)

        # --- Step 4: Refinements ---
        df.loc[
            (df['Cluster'] == 6)
            & (df['dc_is_debit_card_holder'] == 'Yes')
            & (
                (df['debit_card_total_transaction_count_flag'] == 'medium-high')
                | (df['debit_card_total_transaction_amount_flag'].isin(['high', 'medium-high']))
            ),
            'Cluster'
        ] = 2

        df.loc[
            (df['Cluster'] == 6)
            & (df['has_credit_card'] >= 1)
            & (
                df['credit_card_total_transaction_count_flag'].isin(['high', 'medium-high'])
                | (df['credit_card_total_transaction_amount_flag'].isin(['high', 'medium-high']))
            ),
            'Cluster'
        ] = 1

        # --- Step 5: Save labels ---
        self.labels_ = df['Cluster'].values
        self.cluster_names_ = df['Cluster'].unique().tolist()

        return self

    def predict(self, df: pd.DataFrame):
        """Return the assigned cluster labels for the given DataFrame"""
        return df['Cluster'].values

    def fit_predict(self, df: pd.DataFrame):
        """Shortcut: fit the model and return labels"""
        self.fit(df)
        return self.labels_