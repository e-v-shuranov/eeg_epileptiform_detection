class prop:
    def __init__(self):
        self.ii = 0
        self.count_of_nodes = 0
        self.dict_of_nodes = {}
        self.matrix_of_nodes = []

class node:
    def __init__(self, list_of_nodes):
        self.nodes=[]
        for i in list_of_nodes:
            self.nodes.append(i)
    def add_node(self, new_node):
        self.nodes.append(new_node)

    def go_to_next_node(self,cur_node, prop):
        if cur_node in prop.dict_of_nodes:
            return 0
        prop.dict_of_nodes[cur_node] = prop.count_of_nodes
        prop.count_of_nodes += 1

        for i in cur_node.nodes:
            self.go_to_next_node(i,prop)

    def update_matrix_with_ones(self, cur_node, prop):
        if not cur_node in prop.dict_of_nodes:
            return 0
        cur_pos = prop.dict_of_nodes[cur_node]
        prop.dict_of_nodes.pop(cur_node)
        for i in cur_node.nodes:
            if (not i == None) and i in prop.dict_of_nodes:
                prop.matrix_of_nodes[prop.dict_of_nodes[i]][cur_pos] = 1
                prop.matrix_of_nodes[cur_pos][prop.dict_of_nodes[i]] = 1
                if not self is i:
                    self.update_matrix_with_ones(i,prop)

prop = prop()
x = node([])
y=node([])
z=node([])
x.add_node(y)
y.add_node(x)
y.add_node(z)
z.add_node(y)
x.go_to_next_node(x,prop)

#prop.matrix_of_nodes =  [[0]*(prop.count_of_nodes)] *(prop.count_of_nodes)  - ERROR it will be list of the same lines!
prop.matrix_of_nodes = [ [0]*(prop.count_of_nodes) for _ in range((prop.count_of_nodes)) ]
x.update_matrix_with_ones(x,prop)
print(prop.matrix_of_nodes)