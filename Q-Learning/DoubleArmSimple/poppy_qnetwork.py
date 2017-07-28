'''
class for deep neural network execution:
the code is very simple python code and not based on any library (neither numpy)
in order to be fully compatible with any setup
'''

QNETWORK_NAME = "online_network"

class QNetwork:
    def __init__(self, network_name=QNETWORK_NAME):
        self.network = self.load_network(network_name)

    def add(self, m1, m2):
        # add bias
        for i in range(len(m2)):
            m1[0][i] += m2[i][0]
        return m1

    def dot(self, m1, m2):
        # matrix & vector multiplications
        prodM = []
        for i in range(len(m1)): #for each row of m1
            row = m1[i]
            newRow = []
            for j in range(len(m2[0])): #for each column of m2
                y = 0
                for x in range(len(row)):
                    rowEl = row[x]
                    colEl = m2[x][j]
                    y += rowEl*colEl
                newRow.append(y)
            prodM.append(newRow)
        return prodM

    def load_network(self, network_name):
        net = []
        for l in range(8):
            file = open(network_name+'/'+network_name+'%d.txt' % l, 'r')
            Layer = file.read().split('\n')[:-1]
            net.append( [map(float,Layer[i].split()) for i in range(len(Layer))] )
        return net

    def predict(self, s1):
        s2 = self.relu(self.add(self.dot([s1], self.network[0]), self.network[1]))
        s3 = self.relu(self.add(self.dot(s2,self.network[2]), self.network[3]))
        s4 = self.relu(self.add(self.dot(s3,self.network[4]), self.network[5]))
        return self.add(self.dot(s4,self.network[6]), self.network[7])

    def relu(self, m):
        # max(x,0) operation
        for i in range(len(m[0])):
            m[0][i] = max(m[0][i],0)
        return m