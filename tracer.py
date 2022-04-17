import json
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

class Tracer():

    def __init__(self, name, instances):

        self.name = name
        self.instances = instances
        self.mode={'train':MeanCounters(instances),'test':MeanCounters(instances)}
        self.train_data=[]
        self.test_data=[]

    def snap_shots(self,epoch, T=1):
        self.train_data.append(self.mode['train'].snap_shots(epoch))
        if epoch%T==0: self.test_data.append(self.mode['test'].snap_shots(epoch))

    def plot(self):
        train = pd.DataFrame(self.train_data)
        test = pd.DataFrame(self.test_data)

        for obj in self.instances:
            fig, ax = plt.subplots()
            ax.plot(train["epoch"].tolist(),train[obj].tolist(),label=f"Train")
            ax.plot(test["epoch"].tolist(),test[obj].tolist(),label=f"Test")
            ax.set_xlabel('Epoch')
            ax.set_ylabel(obj)
            ax.legend()
            plt.savefig(f"{self.name}_{obj}.png")

    def reset(self):
        self.mode['train'].reset()
        self.mode['test'].reset()

    def load(self):

        try:

            with open(f"{self.name}_train.json", 'r', encoding='utf-8') as f:
                self.train_data = json.load(f)
                
            with open(f"{self.name}_test.json", 'r', encoding='utf-8') as f:
                self.test_data = json.load(f)

            epoch = self.train_data[-1]["epoch"]

        except:

            self.train_data=[]
            self.test_data=[]
            epoch = 0

        return epoch

    def save(self):

        with open(f"{self.name}_train.json", 'w', encoding='utf-8') as f:
            json.dump(self.train_data, f, ensure_ascii=False, indent=4)

        with open(f"{self.name}_test.json", 'w', encoding='utf-8') as f:
            json.dump(self.test_data, f, ensure_ascii=False, indent=4)

class MeanCounters():
    def __init__(self, instances):
        self.instances = instances
        self.counter={}
        for instance in instances:
            self.counter[instance] = MeanCounter()

    def reset(self):
        for obj in self.counter:
            self.counter[obj].reset()

    def snap_shots(self,epoch):

        img={"epoch":epoch}
        for instance in self.instances:
            img[instance] = self.counter[instance].out()

        return img

class MeanCounter():

    def __init__(self):
        self.reset()

    def reset(self):
        self.x=0
        self.n=0
                
    def add(self,x, n=1):
        self.n += n
        self.x += x

    def out(self):
        return self.x / self.n 