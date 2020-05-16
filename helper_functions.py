from pomegranate import State, HiddenMarkovModel, DiscreteDistribution

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

def create_graph(q_df,e_df):
    """
    Creates a markov graph based on tran. dataframe and the emission framework
    and saves the result as markov_graph.dot
    @args:
    - q_df(pd.Dataframe): trans. dataframe
    - e_df(pd.Dataframe): emission dataframe
    return:
        nothing
    """
    '#1.Step: Create graph object'
    G = nx.MultiDiGraph()
    '#2.Step: Add the possible states'
    G.add_nodes_from(q_df.keys())

    '#3.Step: Define colors for the edges. The sum of the edges for each color need to be 1!'
    color=['black', ' grey', ' magenta', ' red', ' blue',
            ' green', ' brown', ' pink', ' orange', ' purple']
    i=0
    '#4.Step: Add the edges for transition probabilities '
    for key,item in q_df.to_dict("index").items():
        for item_name,value in item.items():
            print(key, " , ",item_name,": ",value)
            tmp_origin, tmp_destination = key,item_name
            G.add_edge(tmp_origin, tmp_destination, weight=value, label=value,color=color[i],
                   fontcolor=color[i])
        i+=1     

    '#5.Step: Add the notes for hidden_states '   
    G.add_nodes_from(e_df.index,color="purple",fontcolor="purple")
    '#6.Step: Add the edges for emission probabilities '
    for key,item in e_df.to_dict().items():

        for item_name,value in item.items():
            print(key, " , ",item_name,": ",value)
            tmp_origin, tmp_destination = key,item_name
            G.add_edge(tmp_origin, tmp_destination, weight=value, label=value,color=color[i],
                   fontcolor=color[i])
        i+=1    
    '#7.Step: Create a graph based on G '    
    pos = nx.drawing.nx_pydot.graphviz_layout(G, prog='dot')
    # create edge labels for jupyter plot but is not necessary
    edge_labels = {(n1,n2):d['label'] for n1,n2,d in G.edges(data=True)}
    '#8.Step: Save it to the '    
    nx.drawing.nx_pydot.write_dot(G, 'markov_graph.dot')



def create_hidden_MarkovModel(e_df,q_df,start_p_dict):
    """
    Creates a Hidden Markov Model based on DataFrame
    @args:
        - e_df (pd.Dataframe): contains the emission probabilites
        - q_df (pd.Dataframe): contains the emission probabilites
    """
    model = HiddenMarkovModel(name="Example Model")

    '#1: Create a dict for each key in trans. df'
    model_dict={}
    for key in q_df.keys().values:
        model_dict[key]={}

    '#2: Create the states'  
    for key in model_dict:
        '#2.1.Step Add teh emission prob. to each state, , P(observation | state)'
        emission_p = DiscreteDistribution(e_df[key].to_dict())
        sunny_state = State(emission_p,name=key)
        model_dict[key]=State(emission_p,name=key)
        model.add_state(model_dict[key])
        '#2.2.Step: Add the start probability for each state'
        model.add_transition(model.start, model_dict[key], start_p_dict[key])

    '#3.Step: Add the transition probability to each state'
    for key,item in q_df.to_dict("index").items():
        for item_name,value in item.items():
                print(key, " , ",item_name,": ",value)
                tmp_origin = model_dict[key]
                tmp_destination = model_dict[item_name]
                model.add_transition(tmp_origin, tmp_destination, q_df.loc[key,item_name]) 
    # finally, call the .bake() method to finalize the model
    model.bake()
    
    return model
def preddict_viterbi(model, observations):
    """
    Prints the most likey sequence based on  observations
    @args:
        model -  class HiddenMarkovModel(pomegranate.base.GraphModel)
        
    """
    viterbi_likelihood, viterbi_path = model.viterbi(observations)

    print("The most likely weather sequence to have generated " + \
          "these observations is {} at {:.5f}%."
          .format([s[1].name for s in viterbi_path[1:]], np.exp(viterbi_likelihood)*100)
    )
