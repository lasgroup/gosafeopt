from rich import print

class Logger:
    class_var = {"verbosity" : 0}

    @staticmethod
    def setVerbosity(i):
        Logger.class_var["verbosity"] = i

    @staticmethod
    def warn(msg):
        if Logger.class_var["verbosity"] >= 2:
            print(f"[yellow][WARNING][/yellow] {msg}")

    @staticmethod
    def info(msg):
        if Logger.class_var["verbosity"] >= 3:
            print(f"[white][INFO][/white] {msg}")

    @staticmethod
    def err(msg):
        if Logger.class_var["verbosity"] >= 1:
            print(f"[red][ERROR][/red] {msg}")

    @staticmethod
    def success(msg):
        if Logger.class_var["verbosity"] >= 4:
            print(f"[green][SUCCESS][/green] {msg}")

