class DataLogger:
    def __init__(self, config, context=None):
        self.trajectory_buffer = []
        self.model_buffer = []
        self.reward_max_buffer = []
        self.loss_aq_buffer = []
        self.backup_triggered_buffer = []

        self.x_buffer = []
        self.y_buffer = []

        self.c = config

        self.i = 0

        self.context = context

    def log(self, model, trajectory, x, y, data, rewardMax, loss_aq, backup_triggered, episode, info):
        if self.c["log_plots"]:
            self.trajectory_buffer.append(trajectory)
            self.model_buffer.append(model.state_dict())

        self.backup_triggered_buffer.append(backup_triggered)
        self.loss_aq_buffer.append(loss_aq)
        self.x_buffer.append(x)
        self.y_buffer.append(y)
        self.data = data

        self.reward_max_buffer.append(rewardMax[0])

        self.i += 1

    def getDataFromEpoch(self, i):
        model = create_model(self.c, self.data, self.model_buffer[i])
        return [
            model,
            self.trajectory_buffer[: i + 1],
            torch.reshape(torch.cat(self.x_buffer), (-1, self.c["dim"])),
            torch.reshape(torch.cat(self.y_buffer), (-1, self.c["dim_obs"])),
            self.reward_max_buffer[: i + 1],
            self.backup_triggered_buffer[: i + 1],
        ]

    def save(self, path):
        try:
            with open(path, "wb") as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as ex:
            print("Error during pickling object (Possibly unsupported):", ex)


def load(filename):
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except Exception as ex:
        print("Error during unpickling object (Possibly unsupported):", ex)
