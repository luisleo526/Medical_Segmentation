from subprocess import Popen, PIPE
import visdom
import sys
import pandas as pd

class Visualizer():

    def __init__(self, args):

        self.port = args.display_port
        self.vis  = visdom.Visdom(server="http://localhost", port=self.port, env="main")
        if not self.vis.check_connection():
            self.create_visdom_connections()


    def create_visdom_connections(self):
        cmd = sys.executable + ' -m visdom.server -p %d &>/dev/null &' % self.port
        print('\n\nCould not connect to Visdom server. \n Trying to start a server....')
        print('Command: %s' % cmd)
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    def plot(self, tracer):

        try:

            train = pd.DataFrame(tracer.train_data)
            test = pd.DataFrame(tracer.test_data)

            for args in tracer.instances:
                self.vis.line(X=train["epoch"].tolist(), Y=train[args].tolist(), win=args, name="train", update=None, 
                    opts=dict(xlabel="Epoch", ylabel=args, showlegend=True) )

                if "epoch" in test.columns:
                    self.vis.line(X=test["epoch"].tolist(), Y=test[args].tolist(), win=args, name="test", update="append")

        except:

            self.create_visdom_connections()




