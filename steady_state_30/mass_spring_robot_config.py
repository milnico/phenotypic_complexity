import numpy as np

def mutate_this(gene_to_mut,genotype):

    if gene_to_mut == 0:
        len_seg = np.random.uniform(low=0.05, high=0.12)
        angle = genotype[1]  # np.random.uniform(low=-1, high=1) * np.pi
        x_seg = np.round(len_seg * np.cos(angle), 2)
        y_seg = np.round(len_seg * np.sin(angle), 2)

        actuation = genotype[2]
        s = genotype[3]
        ph = genotype[-1]
    if gene_to_mut == 1:
        len_seg = genotype[0]
        angle = np.random.uniform(low=-1, high=1) * np.pi
        x_seg = np.round(len_seg * np.cos(angle), 2)
        y_seg = np.round(len_seg * np.sin(angle), 2)
        actuation = genotype[2]
        s = genotype[3]
        ph = genotype[-1]
    if gene_to_mut == 2:
        len_seg = genotype[0]
        angle = genotype[1]  # np.random.uniform(low=-1, high=1) * np.pi
        x_seg = np.round(len_seg * np.cos(angle), 2)
        y_seg = np.round(len_seg * np.sin(angle), 2)
        actuation = np.random.uniform(low=0.0, high=0.1)
        s = genotype[3]
        ph = genotype[-1]
    if gene_to_mut == 3:
        len_seg = genotype[0]
        angle = genotype[1]  # np.random.uniform(low=-1, high=1) * np.pi
        x_seg = np.round(len_seg * np.cos(angle), 2)
        y_seg = np.round(len_seg * np.sin(angle), 2)
        actuation = genotype[2]
        s = np.random.randint(10000, 20000)
        ph = genotype[-1]
    if gene_to_mut == 4:
        len_seg = genotype[0]
        angle = genotype[1]  # np.random.uniform(low=-1, high=1) * np.pi
        x_seg = np.round(len_seg * np.cos(angle), 2)
        y_seg = np.round(len_seg * np.sin(angle), 2)
        actuation = genotype[2]
        s = genotype[3]
        ph = np.random.uniform(low=-1.0, high=1.0)
    return len_seg, angle, x_seg,y_seg,s,actuation,ph



class Morph(object):
    def __init__(self):
        self.objects = []
        self.springs = []
        self.create_morph()

    def add_object(self,x):
        self.objects.append(x)
        return len(self.objects) - 1

    def add_spring(self, a, b, length=None, stiffness=1, actuation=0.1):
        if length == None:
            length = ((self.objects[a][0] - self.objects[b][0]) ** 2 +
                      (self.objects[a][1] - self.objects[b][1]) ** 2) ** 0.5
            self.springs.append([a, b, length, stiffness, actuation])

    def create_morph(self):
        self.add_object([0.2, 0.1])
        self.add_object([0.3, 0.13])
        self.add_object([0.4, 0.1])
        self.add_object([0.2, 0.2])
        self.add_object([0.3, 0.2])
        self.add_object([0.4, 0.2])

        s = 14000

        def link(a, b, actuation=0.1):
            self.add_spring(a, b, stiffness=s, actuation=actuation)

        link(0, 1)
        link(1, 2)
        link(3, 4)
        link(4, 5)
        link(0, 3)
        link(2, 5)
        link(0, 4)
        link(1, 4)
        link(2, 4)
        link(3, 1)
        link(5, 1)

class Morph_evo(object):
    def __init__(self,first=True):
        self.objects = []
        self.springs = []
        self.store_springs = []
        if first:
            self.create_first_morph()
        else:
            self.createmorph()

    def add_object(self,x):
        self.objects.append(x)
        return len(self.objects) - 1

    def add_mesh_spring(self,a, b, s, act):
        if (a, b) in self.store_springs or (b, a) in self.store_springs or b==a:
            return

        self.store_springs.append((a, b))
        self.add_spring(a, b, stiffness=s, actuation=act)

    def add_spring(self, a, b, length=None, stiffness=1, actuation=0.1):
        if length == None:
            length = ((self.objects[a][0] - self.objects[b][0]) ** 2 +
                      (self.objects[a][1] - self.objects[b][1]) ** 2) ** 0.5
            self.springs.append([a, b, length, stiffness, actuation])

    def create_morph(self):
        s = 14000
        actuation = 0.1
        x = np.linspace(0.1, 0.5, 5)
        y = np.linspace(0.1, 0.5, 5)

        xv, yv = np.meshgrid(x, y)

        for i,j in zip(xv,yv):

            for z,k in zip(i,j):

                self.add_object([round(z,1),round(k,1)])


        springs_link = np.random.randint(0,25,size=(25,2))
        for link in springs_link:
            self.add_mesh_spring(link[0], link[1], s, actuation)


class Morph_evolve(object):
    def __init__(self,first=False):
        #np.random.seed(seed)
        self.objects = []
        self.springs = []
        self.link_id = []
        self.store_springs = []
        self.stored = []
        self.genotype = []

        self.create_morph(first)


    def add_object(self,x):
        self.objects.append(x)
        return len(self.objects) - 1

    def add_mesh_spring(self,a, b, s, act,ph=0.0):

        if ((a, b) in self.stored or (b, a) in self.stored or b==a):
            ##print(self.stored)
            ##print(a,b)
            #input("HERE")
            return

        self.stored.append((a, b))
        self.add_spring(a, b, stiffness=s, actuation=act,phase=ph)

    def add_spring(self, a, b, length=None, stiffness=1, actuation=0.1,phase=0.0):
        if length == None:

            length = ((self.objects[a][0] - self.objects[b][0]) ** 2 +
                      (self.objects[a][1] - self.objects[b][1]) ** 2) ** 0.5
            self.springs.append([a, b, length, stiffness, actuation,phase])


    def create_morph(self,first=False):
        #s = 14000
        #actuation = 0.1
        if first:
            tot_links = 100
        else:
            tot_links = np.random.randint(6,20)
        lk=0
        point_id=0
        max_act = 0.1
        max_stif = 20000
        phase = 1

        while lk <= tot_links:
            if lk == 0:
                self.add_object([0.3, 0.3])
                len_seg = np.random.uniform(low=0.0, high=0.12)
                angle =  np.random.uniform(low=-1, high=1)*np.pi
                x_seg = np.round(len_seg*np.cos(angle),2)
                y_seg = np.round(len_seg*np.sin(angle),2)
                s = np.random.randint(10000,max_stif)
                actuation = np.random.uniform(low=0.0, high=max_act)
                ph = np.random.uniform(low=-phase, high=phase)
                self.add_object([np.round(self.objects[point_id][0] + x_seg,2),np.round(self.objects[point_id][1] + y_seg,2)])
                self.link_id.append([point_id,point_id+1])
                self.store_springs.append((point_id, point_id+1))
                self.genotype.append([len_seg,angle,actuation,s,ph,(point_id, point_id+1)])
                lk+=1
                point_id+=1
            else:
                attach = np.random.uniform(0,1)
                if lk < 4 or attach<0.9:
                    len_seg = np.random.uniform(low=0.0, high=0.12)
                    angle = np.random.uniform(low=-1, high=1) * np.pi
                    x_seg = np.round(len_seg * np.cos(angle), 2)
                    y_seg = np.round(len_seg * np.sin(angle), 2)
                    s = np.random.randint(10000, max_stif)
                    actuation = np.random.uniform(low=0.0, high=max_act)
                    ph = np.random.uniform(low=-phase, high=phase)
                    self.add_object([np.round(self.objects[point_id][0] + x_seg, 2), np.round(self.objects[point_id][1] + y_seg, 2)])
                    self.link_id.append([point_id, point_id + 1])
                    self.store_springs.append((point_id, point_id + 1))
                    self.genotype.append([len_seg, angle, actuation, s, ph, (point_id, point_id + 1)])
                    lk+=1
                    point_id += 1
                else:
                    a,b = point_id,np.random.randint(0,point_id-1)
                    ##print(self.store_springs)
                    attempts = 0
                    while((a,b) in self.store_springs) and attempts<10:

                        a, b = point_id, np.random.randint(0, point_id - 1)
                        attempts+=1

                    if attempts <10:
                        self.store_springs.append((a, b))
                        self.link_id.append([a,b])
                        s = np.random.randint(10000, max_stif)
                        actuation = np.random.uniform(low=0.0, high=max_act)
                        ph = np.random.uniform(low=-phase, high=phase)

                        lk += 1
                        self.genotype.append([0.0, 0.0, actuation, s, ph, (a, b)])
                    else:
                        pass


        for id,link in enumerate(self.link_id):
            ##print(link)
            ##print(self.genotype[id],self.genotype[id][-1])
            self.add_mesh_spring(link[0],link[1],self.genotype[id][3],self.genotype[id][2],self.genotype[id][4])
        #input("eee")
        self.genotype = np.array(self.genotype)
        self.stored=[]


    def mutate_morph(self,mut_Rate=0.02):
        ##print("mutate")
        max_spring=30
        s = 14000
        self.stored = []
        actuation = 0.1
        tot_gene= len(self.genotype)
        #print("prima",self.genotype)
        #print("prima obj",self.objects,len(self.objects))
        #print("prima link",self.link_id,len(self.link_id))
        gene = 0
        point_id = 0
        self.new_genotype = []
        self.new_objects = []
        self.new_springs = []
        self.new_link_id = []
        self.new_store_springs = []
        mutate = False
        max_act = 0.1
        max_stif = 20000
        phase = 1
        self.new_objects.append([0.3,0.3])
        num_mut = 2
        gene_to_mut = np.random.choice(tot_gene, num_mut, replace=False)
        ##print(gene_to_mut)
        #input("mut")
        if tot_gene>max_spring:
            max_spring=tot_gene
        while gene < tot_gene:
            if gene >=max_spring:
                break
            mut = np.random.random()
            if mut < mut_Rate:
                #print("muting")
                mutate=True
                if tot_gene > 30:
                    if np.random.random()<0.5:
                        mutation_type = 0
                    else:
                        mutation_type = 2
                else:
                    mutation_type = np.random.choice(3,1,p=[0.34, 0.33, 0.33])[0]
                    #print("mutation type",mutation_type)

            else:
                mutate=False

            if self.genotype[gene][0]!=0.0:
                #print("gene", self.genotype[gene])
                if mutate == True:
                    #print("gene", self.genotype[gene])
                    if gene < 4 :
                        #print("change mut to zero")
                        mutation_type = 0
                    if mutation_type==0:
                        if point_id == point_id + 1 or (point_id + 1, point_id) in self.new_store_springs or (
                        point_id, point_id + 1) in self.new_store_springs:
                            gene += 1

                        else:
                            #len_seg = np.random.uniform(low=0.0, high=0.12)
                            #angle = self.genotype[gene][1]#np.random.uniform(low=-1, high=1) * np.pi
                            #x_seg = np.round(len_seg * np.cos(angle), 2)
                            #y_seg = np.round(len_seg * np.sin(angle), 2)
                            to_mut = np.random.randint(0,5)
                            len_seg, angle, x_seg, y_seg, s, actuation, ph = mutate_this(to_mut,self.genotype[gene][:-1])

                            self.new_objects.append(
                                [np.round(self.new_objects[point_id][0] + x_seg, 4), np.round(self.new_objects[point_id][1] + y_seg, 4)])
                            self.new_link_id.append([point_id, point_id + 1])
                            self.new_store_springs.append((point_id, point_id + 1))
                            self.new_genotype.append([len_seg, angle, actuation, s, ph, (point_id, point_id + 1)])
                            gene += 1
                            point_id += 1


                    if mutation_type==1:
                        #first put the existing connection
                        if point_id == point_id + 1 or (point_id + 1, point_id) in self.new_store_springs or (
                        point_id, point_id + 1) in self.new_store_springs:
                            gene += 1
                        else:
                            len_seg = self.genotype[gene][0]  # np.random.uniform(low=0.0, high=0.12)
                            angle = self.genotype[gene][1]  # np.random.uniform(low=-1, high=1) * np.pi
                            x_seg = np.round(len_seg * np.cos(angle), 2)
                            y_seg = np.round(len_seg * np.sin(angle), 2)
                            s = self.genotype[gene][3]
                            actuation = self.genotype[gene][2]
                            ph = self.genotype[gene][4]
                            self.new_objects.append(
                                [np.round(self.new_objects[point_id][0] + x_seg, 4),
                                 np.round(self.new_objects[point_id][1] + y_seg, 4)])
                            self.new_link_id.append([point_id, point_id + 1])
                            self.new_store_springs.append((point_id, point_id + 1))
                            self.new_genotype.append([len_seg, angle, actuation, s, ph, (point_id, point_id + 1)])
                            point_id +=1
                            #then add a new one
                            if np.random.random()<0.5:
                               # #print("HEREEEEEE")
                                a, b = point_id, np.random.randint(0, point_id - 1)
                                ##print(self.store_springs)
                                attempts = 0
                                while ((a, b) in self.new_store_springs or (b, a) in self.new_store_springs or b==a) and attempts < 10:
                                    a = point_id
                                    b = np.random.randint(0, point_id - 2)
                                    attempts += 1
                                #print("tttt",attempts)
                                if attempts < 10:
                                    #print("HERE1")

                                    self.new_store_springs.append((a, b))
                                    self.new_link_id.append([a, b])
                                    s = np.random.randint(10000, max_stif)
                                    actuation = np.random.uniform(low=0.0, high=max_act)
                                    ph = np.random.uniform(low=-phase, high=phase)
                                    gene += 1
                                    self.new_genotype.append([0.0, 0.0, actuation, s, ph, (a, b)])
                                else:
                                    #print("GGGGGG1333")
                                    gene+=1
                            else:
                                if point_id == point_id + 1 or (point_id + 1, point_id) in self.new_store_springs or (
                                        point_id, point_id + 1) in self.new_store_springs:

                                    gene+=1
                                else:
                                    ##print("HEREEEEEE2")
                                    len_seg = self.genotype[gene][0]  # np.random.uniform(low=0.0, high=0.12)
                                    angle = self.genotype[gene][1]  # np.random.uniform(low=-1, high=1) * np.pi
                                    x_seg = np.round(len_seg * np.cos(angle), 2)
                                    y_seg = np.round(len_seg * np.sin(angle), 2)
                                    s = np.random.randint(10000, max_stif)
                                    actuation = np.random.uniform(low=0.0, high=max_act)
                                    ph = np.random.uniform(low=-phase, high=phase)
                                    self.new_objects.append(
                                        [np.round(self.new_objects[point_id][0] + x_seg, 4),
                                         np.round(self.new_objects[point_id][1] + y_seg, 4)])
                                    self.new_link_id.append([point_id, point_id + 1])
                                    self.new_store_springs.append((point_id, point_id + 1))
                                    self.new_genotype.append([len_seg, angle, actuation, s, ph, (point_id, point_id + 1)])
                                    point_id += 1
                                    gene += 1



                    if mutation_type==2:
                        ##print("gene to delete",self.genotype[gene])
                        gene += 1


                else:
                    #print("HUHUHUHUHUH", point_id, point_id+1)
                    #print(self.new_store_springs,len(self.new_objects))
                    if point_id == point_id+1 or (point_id+1, point_id) in self.new_store_springs or (point_id, point_id+1) in self.new_store_springs :
                        gene += 1
                    else:
                        len_seg = self.genotype[gene][0]  # np.random.uniform(low=0.0, high=0.12)
                        angle = self.genotype[gene][1]  # np.random.uniform(low=-1, high=1) * np.pi
                        x_seg = np.round(len_seg * np.cos(angle), 2)
                        y_seg = np.round(len_seg * np.sin(angle), 2)
                        s = self.genotype[gene][3]
                        actuation = self.genotype[gene][2]
                        ph = self.genotype[gene][4]
                        self.new_objects.append(
                            [np.round(self.new_objects[point_id][0] + x_seg, 4),
                             np.round(self.new_objects[point_id][1] + y_seg, 4)])
                        self.new_link_id.append([point_id, point_id + 1])
                        self.new_store_springs.append((point_id, point_id + 1))
                        self.new_genotype.append([len_seg, angle, actuation, s, ph, (point_id, point_id + 1)])
                        gene += 1
                    point_id += 1
            else:
                #print("gene",self.genotype[gene])
                if mutate == True:
                    #print("gene", self.genotype[gene])
                    if mutation_type==0:
                        link_1 = point_id  # self.genotype[gene][-1][0]
                        link_2 = self.genotype[gene][-1][1]  # + (point_id-self.genotype[gene][-1][0])
                        if link_1 <= link_2 or (link_2, link_1) in self.new_store_springs or (
                        link_1, link_2) in self.new_store_springs:
                            gene += 1
                        else:
                            # len_seg = np.random.uniform(low=0.0, high=0.12)
                            # angle = self.genotype[gene][1]#np.random.uniform(low=-1, high=1) * np.pi
                            # x_seg = np.round(len_seg * np.cos(angle), 2)
                            # y_seg = np.round(len_seg * np.sin(angle), 2)
                            to_mut = np.random.randint(2, 5)


                            len_seg, angle, x_seg, y_seg, s, actuation, ph = mutate_this(to_mut, self.genotype[gene][:-1])

                            #self.new_objects.append(
                            #    [np.round(self.new_objects[point_id][0] + x_seg, 2), np.round(self.new_objects[point_id][0] + y_seg, 2)])
                            self.new_link_id.append([link_1, link_2])
                            self.new_store_springs.append((link_1, link_2))
                            self.new_genotype.append([0.0, 0.0, actuation, s, ph, (link_1, link_2)])
                            gene += 1
                            #point_id += 1
                    if mutation_type==1:
                        #place existing connection
                        link_1 = point_id  # self.genotype[gene][-1][0]
                        link_2 = self.genotype[gene][-1][1]  # + (point_id-self.genotype[gene][-1][0])
                        if (link_1<=link_2 or (link_1, link_2) in self.new_store_springs or (link_2, link_1) in self.new_store_springs ):
                            
                            gene+=1
                        else:
                            s = self.genotype[gene][3]
                            actuation = self.genotype[gene][2]
                            ph = self.genotype[gene][4]
                            self.new_store_springs.append((link_1, link_2))
                            self.new_link_id.append([link_1, link_2])
                            self.new_genotype.append([0.0, 0.0, actuation, s, ph, (link_1, link_2)])
                            # then add a new one
                            if np.random.random() < 0.5:
                                ##print("HEREEEtttEEE")
                                a, b = point_id, np.random.randint(0, point_id - 2)
                                ##print(self.new_store_springs)
                                attempts = 0
                                while ((a, b) in self.new_store_springs or (b, a) in self.new_store_springs or b==a) and attempts < 10:
                                    ##print((a,b))
                                    a = point_id
                                    b = np.random.randint(0, point_id - 2)
                                    attempts += 1
                                #print("tttt", attempts)
                                if attempts < 10:
                                    #print("HERE2")
                                    self.new_store_springs.append((a, b))
                                    self.new_link_id.append([a, b])
                                    s = np.random.randint(10000, max_stif)
                                    actuation = np.random.uniform(low=0.0, high=max_act)
                                    ph = np.random.uniform(low=-phase, high=phase)
                                    gene += 1
                                    self.new_genotype.append([0.0, 0.0, actuation, s, ph, (a, b)])
                                else:
                                    #print("GGGGGG1")

                                    gene+=1
                            else:
                                if point_id == point_id + 1 or (point_id + 1, point_id) in self.new_store_springs or (
                                        point_id, point_id + 1) in self.new_store_springs:

                                    gene+=1
                                else:
                                    ##print("HEREEEErtrEE2")
                                    len_seg = np.random.uniform(low=0.05, high=0.12)
                                    angle = np.random.uniform(low=-1, high=1) * np.pi
                                    x_seg = np.round(len_seg * np.cos(angle), 2)
                                    y_seg = np.round(len_seg * np.sin(angle), 2)
                                    s = np.random.randint(10000, max_stif)
                                    actuation = np.random.uniform(low=0.0, high=max_act)
                                    ph = np.random.uniform(low=-phase, high=phase)
                                    self.new_objects.append(
                                        [np.round(self.new_objects[point_id][0] + x_seg, 4),
                                         np.round(self.new_objects[point_id][1] + y_seg, 4)])
                                    self.new_link_id.append([point_id, point_id + 1])
                                    self.new_store_springs.append((point_id, point_id + 1))
                                    self.new_genotype.append([len_seg, angle, actuation, s, ph, (point_id, point_id + 1)])
                                    point_id += 1
                                    gene += 1

                    if mutation_type==2:
                        ##print("gene to delete", self.genotype[gene])
                        gene += 1

                else:

                    link_1 = point_id#self.genotype[gene][-1][0]
                    link_2 = self.genotype[gene][-1][1] #+ (point_id-self.genotype[gene][-
                    #print("IAMAMAMAMAMAMA",link_1,link_2)
                    #print(self.new_store_springs)
                    if link_1 <= link_2 or (link_2, link_1) in self.new_store_springs or (link_1, link_2) in self.new_store_springs :
                        gene += 1
                    else:
                        s = self.genotype[gene][3]
                        actuation = self.genotype[gene][2]
                        ph = self.genotype[gene][4]
                        self.new_store_springs.append((link_1,link_2))
                        self.new_link_id.append([link_1, link_2])
                        gene += 1
                        self.new_genotype.append([0.0, 0.0, actuation, s, ph, (link_1, link_2)])



        ##print(self.genotype)
        ##print(np.array(self.new_genotype))
        #input("mut")
        ##print(self.objects)
        ##print(self.new_link_id)
        #input("eee")
        ##print("ff", self.new_objects)
        #tmp_objects = []
        #for ob in self.new_objects:
        #    if ob in tmp_objects:
        #        pass
        #    else:
        #        tmp_objects.append(ob)
        self.objects = self.new_objects
        self.springs = []
        self.link_id = self.new_link_id
        #print("dopo",self.objects,len(self.objects))
        #print("obb",np.array(self.new_genotype))
        #print("link",self.new_link_id,len(self.new_link_id))
        #input("pause")


        #self.new_genotype = [item for item in self.new_genotype if item[-1][0] != len(self.objects)-1]

        for id,link in enumerate(self.new_link_id):

            ##print(id,link)
            ##print([link[1],link[0]] in construct)
            #input("ola")
            #if link[0] == len(self.objects)-1:
            #    pass#del self.new_genotype[id]
            #else:
            #    pass
            #else:
            #try:
            self.add_mesh_spring(link[0],link[1],self.new_genotype[id][3],self.new_genotype[id][2],self.new_genotype[id][-2])
            #except Exception as ex:
             #   #print(id,link)
               # #print(self.objects)
               # input("pause")
            #    construct.append(link)
        #input("fine")

        self.genotype = np.array(self.new_genotype,dtype=object)

    '''
    def mutate_morph(self):
        s = 14000
        actuation = 0.1
        tot_links = len(self.genotype)

        lk=0
        new_link=0
        point_id=0
        self.new_genotype = []
        self.new_objects = []
        self.new_springs = []
        self.new_link_id = []
        self.new_store_springs = []
        mutate = False
        #print(self.genotype)
        while lk < tot_links:

            if np.random.uniform()<-0.02:
                mutate= True
                mutation_type = np.random.choice(3, p=[0.25,0.25,0.5])
            if lk == 0:
                self.new_objects.append([0.3, 0.3])
                if mutate:

                    len_seg = np.random.uniform(low=0.0, high=0.12)
                    angle =  np.random.uniform(low=-1, high=1)*np.pi
                    x_seg = np.round(len_seg*np.cos(angle),2)
                    y_seg = np.round(len_seg*np.sin(angle),2)
                    self.new_objects.append([np.round(self.objects[point_id][0] + x_seg,2),np.round(self.objects[point_id][0] + y_seg,2)])
                    self.new_link_id.append([point_id,point_id+1])
                    self.new_store_springs.append((point_id, point_id+1))
                    self.new_genotype.append([len_seg,angle,actuation,s,0,(point_id, point_id+1)])
                else:
                    self.new_genotype.append(self.genotype[lk])
                    self.new_objects.append(self.objects[point_id+1])
                    self.new_link_id.append(self.link_id[lk])
                    self.new_store_springs.append(self.store_springs[lk])

                lk+=1
                new_link+=1
                point_id+=1
            else:
                #attach = np.random.uniform(0,1)
                if lk < 4:
                    if mutate:

                        len_seg = np.random.uniform(low=0.0, high=0.12)
                        angle =  np.random.uniform(low=-1, high=1)*np.pi
                        x_seg = np.round(len_seg*np.cos(angle),2)
                        y_seg = np.round(len_seg*np.sin(angle),2)
                        self.new_objects.append([np.round(self.objects[point_id][0] + x_seg,2),np.round(self.objects[point_id][0] + y_seg,2)])
                        self.new_link_id.append([point_id,point_id+1])
                        self.new_store_springs.append((point_id, point_id+1))
                        self.new_genotype.append([len_seg,angle,actuation,s,0,(point_id, point_id+1)])
                    else:
                        self.new_genotype.append(self.genotype[lk])
                        self.new_objects.append(self.objects[point_id+1])
                        self.new_link_id.append(self.link_id[lk])
                        self.new_store_springs.append(self.store_springs[lk])

                    lk+=1
                    new_link += 1
                    point_id+=1
                else:
                    if mutate:
                        if mutation_type == 0:
                            len_seg = np.random.uniform(low=0.0, high=0.12)
                            angle = np.random.uniform(low=-1, high=1) * np.pi
                            x_seg = np.round(len_seg * np.cos(angle), 2)
                            y_seg = np.round(len_seg * np.sin(angle), 2)
                            self.new_objects.append([np.round(self.objects[point_id][0] + x_seg, 2),
                                                     np.round(self.objects[point_id][0] + y_seg, 2)])
                            self.new_link_id.append([point_id, point_id + 1])
                            self.new_store_springs.append((point_id, point_id + 1))
                            self.new_genotype.append([len_seg, angle, actuation, s, 0, (point_id, point_id + 1)])
                            lk += 1
                            new_link += 1
                            point_id += 1
                        if mutation_type == 1:#add connection to existing node
                            a, b = point_id, np.random.randint(0, point_id - 1)
                            #print(self.store_springs)
                            attempts = 0
                            while ((a, b) in self.new_store_springs) and attempts < 10:
                                a, b = point_id, np.random.randint(0, point_id - 1)
                                attempts += 1

                            if attempts < 10:
                                self.new_store_springs.append((a, b))
                                self.new_link_id.append([a, b])
                                self.new_genotype.append([0.0, 0.0, actuation, s, 1, (a, b)])
                                new_link += 1
                            else:
                                pass
                        if mutation_type == 2:#delete connection
                            lk+=1
                            new_link += 1
                            point_id += 1
                    else:
                        self.new_genotype.append(self.genotype[lk])
                        self.new_objects.append(self.objects[point_id + 1])
                        self.new_link_id.append(self.link_id[lk])
                        self.new_store_springs.append(self.store_springs[lk])
                        lk += 1
                        new_link += 1
                        point_id += 1




        #print(self.objects)
        #print(self.new_objects)
        
        input("ee")
        for link in self.link_id:

            self.add_mesh_spring(link[0],link[1],s,actuation)

    '''




class Morph_2(object):
    def __init__(self):
        self.objects = []
        self.springs = []
        self.store_springs = []
        self.create_morph()

    def add_object(self,x):
        self.objects.append(x)
        return len(self.objects) - 1

    def add_mesh_spring(self,a, b, s, act):
        if (a, b) in self.store_springs or (b, a) in self.store_springs or b==a:
            return

        self.store_springs.append((a, b))
        self.add_spring(a, b, stiffness=s, actuation=act)

    def add_spring(self, a, b, length=None, stiffness=1, actuation=0.1):
        if length == None:
            length = ((self.objects[a][0] - self.objects[b][0]) ** 2 +
                      (self.objects[a][1] - self.objects[b][1]) ** 2) ** 0.5
            self.springs.append([a, b, length, stiffness, actuation])

    def create_morph(self):
        s = 14000
        actuation = 0.1
        x = np.linspace(0.2, 0.4, 3)
        y = np.linspace(0.1, 0.2, 2)

        xv, yv = np.meshgrid(x, y)

        for i,j in zip(xv,yv):

            for z,k in zip(i,j):

                self.add_object([round(z,1),round(k,1)])


        springs_link = np.random.randint(0,6,size=(11,2))
        for link in springs_link:
            self.add_mesh_spring(link[0], link[1], s, actuation)

        ##print(self.springs)
        ##print(self.objects)
        #input("eej")

objects = []
springs = []


def add_object(x):
    objects.append(x)
    return len(objects) - 1


def add_spring(a, b, length=None, stiffness=1, actuation=0.1):
    if length == None:
        length = ((objects[a][0] - objects[b][0])**2 +
                  (objects[a][1] - objects[b][1])**2)**0.5
    springs.append([a, b, length, stiffness, actuation])


def robotA():
    input("robat 0")
    add_object([0.2, 0.1])
    add_object([0.3, 0.13])
    add_object([0.4, 0.1])
    add_object([0.2, 0.2])
    add_object([0.3, 0.2])
    add_object([0.4, 0.2])

    s = 14000

    def link(a, b, actuation=0.1):
        add_spring(a, b, stiffness=s, actuation=actuation)

    link(0, 1)
    link(1, 2)
    link(3, 4)
    link(4, 5)
    link(0, 3)
    link(2, 5)
    link(0, 4)
    link(1, 4)
    link(2, 4)
    link(3, 1)
    link(5, 1)
    

    return objects, springs


points = []
point_id = []
mesh_springs = []


def add_mesh_point(i, j):
    if (i, j) not in points:
        id = add_object((i * 0.05 + 0.1, j * 0.05 + 0.1))
        points.append((i, j))
        point_id.append(id)
    return point_id[points.index((i, j))]


def add_mesh_spring(a, b, s, act):
    if (a, b) in mesh_springs or (b, a) in mesh_springs:
        return

    mesh_springs.append((a, b))
    add_spring(a, b, stiffness=s, actuation=act)


def add_mesh_square(i, j, actuation=0.0):
    a = add_mesh_point(i, j)
    b = add_mesh_point(i, j + 1)
    c = add_mesh_point(i + 1, j)
    d = add_mesh_point(i + 1, j + 1)

    # b d
    # a c
    add_mesh_spring(a, b, 3e4, actuation)
    add_mesh_spring(c, d, 3e4, actuation)

    for i in [a, b, c, d]:
        for j in [a, b, c, d]:
            if i != j:
                add_mesh_spring(i, j, 3e4, 0)


def add_mesh_triangle(i, j, actuation=0.0):
    a = add_mesh_point(i + 0.5, j + 0.5)
    b = add_mesh_point(i, j + 1)
    d = add_mesh_point(i + 1, j + 1)

    for i in [a, b, d]:
        for j in [a, b, d]:
            if i != j:
                add_mesh_spring(i, j, 3e4, 0)


def robotB():
    add_mesh_triangle(2, 0, actuation=0.15)
    add_mesh_triangle(0, 0, actuation=0.15)
    add_mesh_square(0, 1, actuation=0.15)
    add_mesh_square(0, 2)
    add_mesh_square(1, 2)
    add_mesh_square(2, 1, actuation=0.15)
    add_mesh_square(2, 2)
    # add_mesh_square(2, 3)
    # add_mesh_square(2, 4)

    return objects, springs


def robotC():
    add_mesh_square(2, 0, actuation=0.15)
    add_mesh_square(0, 0, actuation=0.15)
    add_mesh_square(0, 1, actuation=0.15)
    add_mesh_square(0, 2)
    add_mesh_square(1, 2)
    add_mesh_square(2, 1, actuation=0.15)
    add_mesh_square(2, 2)
    add_mesh_square(2, 3)
    add_mesh_square(2, 4)

    return objects, springs


def robotD():
    add_mesh_square(2, 0, actuation=0.15)
    add_mesh_square(0, 0, actuation=0.15)
    add_mesh_square(0, 1, actuation=0.15)
    add_mesh_square(0, 2)
    add_mesh_square(1, 2)
    add_mesh_square(2, 1, actuation=0.15)
    add_mesh_square(2, 2)
    add_mesh_square(2, 3)
    add_mesh_square(2, 4)
    add_mesh_square(3, 1)
    add_mesh_square(4, 0, actuation=0.15)
    add_mesh_square(4, 1, actuation=0.15)

    return objects, springs


robots = [robotA, robotB, robotC, robotD]
