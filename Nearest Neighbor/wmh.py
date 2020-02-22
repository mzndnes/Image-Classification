import numpy as np
import random
from operator import itemgetter

class wmh_hash:
    def __init__(self,train,test,h,k):
        self._h=h
        self._train=train
        self._test=test
        self._k=k
        seed = np.zeros((h), dtype=np.int)
        for i in range(h):
            seed[i] = random.randint(1, 10000)
        self._seed=seed



        trinst = len(train)
        tinst = len(test)
        dim = len(train[0]) - 1

        maxi = np.zeros(dim, dtype=np.float)

        for i in range(tinst):
            inte = test[i][1:]
            for j in range(dim):
                maxi[j] = max(maxi[j], inte[j])

        for i in range(trinst):
            inte = train[i][1:]

            for j in range(dim):
                maxi[j] = max(maxi[j], inte[j])

        # print(maxi)
        for i in range(len(maxi)):
            if maxi[i] <= 0:
                maxi[i] = 1

        maxx = np.ceil(maxi)

        M = 0
        m = np.zeros(len(maxi) + 1, dtype=np.int)

        for i in range(len(maxi)):
            m[i] = M
            M += maxx[i]

        m[i + 1] = M

        mp = np.zeros(int(M), dtype=np.int)
        for i in range(len(m) - 1):
            for j in range(m[i], m[i + 1]):
                mp[j] = i
        self._m=m
        self._mp=mp
        self._M=M

    def gen_hash(self,v):
        set_hs=[]
        for j in range(len(v)):
            hs=[]
            for i in range(self._h):
                mn = 0
                generator = np.random.RandomState(self._seed[i])
                while True:
                    u = generator.uniform(0, self._M, 1).astype(np.float)
                    idx = int(np.floor(u))

                    indx1 = self._mp[idx]
                    if u[0] <= self._m[indx1] + v[j][indx1]:
                        break

                    mn += 1
                hs.append(mn)
            set_hs.append(hs)
        return set_hs

    def memorize(self,trhash):
        self._trhash=trhash


    def collision(self,v1,v2):
        intersection = 0
        for k in range(len(v1)):
            if v1[k] == v2[k]:
                intersection += 1
        return intersection

    def predict(self,t):
        esim =[]
        trinst = len(self._trhash)
        for j in range(trinst):
            trsim = []
            trsim.append(j)
            trsim.append(self.collision(t, self._trhash[j]) / self._h)
            esim.append(list(trsim))


        esim.sort(key=itemgetter(1))

        cnt={}
        for j in range(self._k):
            trnum=int(esim[j][0])
            if int(self._train[trnum][0]) in cnt:  # onnidxs after checking all train
                cnt[self._train[j][0]]+=1
            else:
                cnt[self._train[j][0]]=1
        mx=-1

        for k in cnt.keys():
            if cnt[k]>mx:
                mx=cnt[k]
                ky=k
        return ky