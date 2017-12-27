import csv
import matplotlib.pyplot as plt

class ResultsAnalyzer:

    def load(self):
        data = []
        with open(self.results_csv_file_path, 'r') as f:
            reader = csv.reader(f)
            headers = None
            for row in reader:
                if headers is None:
                    headers = row
                else:
                    datum = {}
                    for col, text in enumerate(row):
                        datum[headers[col]] = text
                    data.append(datum)
        self.data = data

    def __init__(self, results_csv_file_path,
                 execution_id_header='Execution ID',
                 score_header='Tuner Score',
                 scope_header='Scope',
                 class_header='Class',
                 epoch_header='Epoch',
                 last_epoch_header='Last Epoch',
                 best_epoch_header='Best Epoch'
                 ):
        self.best_epoch_header = best_epoch_header
        self.last_epoch_header = last_epoch_header
        self.class_header = class_header
        self.epoch_header = epoch_header
        self.scope_header = scope_header
        self.execution_id_header = execution_id_header
        self.score_header = score_header
        self.data = None
        self.executions = None
        self.results_csv_file_path = results_csv_file_path
        self.load()
        self.process_executions()

    def process_executions(self):
        self.executions = {}
        for row in self.data:
            if row[self.scope_header] != 'test' \
                or row[self.class_header] != '-- All Classes --' \
                or row[self.best_epoch_header] != 'Yes':
                    continue

            self.executions[row[self.execution_id_header]] = row

    def get_executions_count(self):
        return len(self.executions)

    def plot(self, x, y, filter=None):
        filter = filter or (lambda x: True)

        def numerize(values):
            try:
                values = [float(x) for x in values]
                return values, None
            except ValueError as e:
                values_set = sorted(set(values))
                return [values_set.index(v) for v in values], values_set

        points = [(e[x], e[y]) for e in self.executions.values() if filter(e)]
        x_values, x_tick_labels = numerize([p[0] for p in points])
        y_values, y_tick_labels = numerize([p[1] for p in points])
        plt.scatter(x_values, y_values)
        # if x_tick_labels:
        #     plt.xticks(x_tick_labels)
        # if y_tick_labels:
        #     plt.xticks(y_tick_labels)
        plt.xlabel(x)
        plt.ylabel(y)
        plt.show()

    def plot_on_score(self, x, filter=None):
        self.plot(x=x, y=self.score_header, filter=filter)

    def plot_score_dist(self, bin_size=5):
        scores = [e[self.score_header] for e in self.executions.values()]
        dist = {}
        for s in scores:
            bin = float(s) - float(s) % bin_size
            dist[bin] = dist.get(bin, 0) + 1
        points = dist.items()
        x_values = [p[0] for p in points]
        y_values = [p[1] for p in points]
        plt.scatter(x_values, y_values)
        plt.xlabel('score')
        plt.xlabel('count')
        plt.show()



if __name__ == '__main__':
    an = ResultsAnalyzer('/cs/usr/aviramstern/lab/results (copy).csv')
    filter = lambda e: e["learning_rate"] == '0.1'
    an.plot_on_score('learning_rate', filter=filter)
    an.plot_on_score('learning_rate_decay', filter=filter)
    an.plot_on_score('use_head', filter=filter)
    an.plot_on_score('use_dep', filter=filter)
    an.plot_on_score('use_prep_onehot', filter=filter)
    an.plot_on_score('use_token_internal', filter=filter)
    an.plot_on_score('token_internal_embd_dim', filter = lambda e: e["use_token_internal"] == 'True'and filter(e))
    an.plot_on_score('mlp_layers', filter=filter)
    an.plot_on_score('mlp_layer_dim', filter=filter)
    an.plot_on_score('mlp_activation', filter=filter)
    an.plot_on_score('lstm_h_dim', filter=filter)
    an.plot_on_score('mlp_dropout_p', filter=filter)
    an.plot_on_score('update_token_embd', filter=filter)
    an.plot_score_dist()
