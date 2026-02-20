class Metrics:
    @staticmethod
    def get_logit_diff(true_label_i, distorted_label_i):
        def metric(logits):
            return (logits[:, distorted_label_i] - logits[:, true_label_i]).mean()
        return metric

    @staticmethod
    def get_logit_diff_tp(true_label_i, distorted_label_i):
        def metric(logits_clean, logits_distorted):
            return (logits_clean[:, distorted_label_i] + logits_distorted[:, true_label_i]
                    - logits_clean[:, true_label_i] - logits_distorted[:, distorted_label_i]).mean()

        return metric