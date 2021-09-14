import numpy as np
import pandas as pd
import random
import itertools as it

# Argument pasers
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--no_of_genres', type=int, help='no of genres in data: should be <=25', default=11)
parser.add_argument('--alpha', type=int, help='reward weightage', default=0.8)
parser.add_argument('--gamma', type=int, help='weightage to future reward or discount factor', default=0.4)
parser.add_argument('--epsilon', type=int, help='exploration probability or epsilon greedy value', default=0.3)
optim = parser.parse_args()

# Data acquisition and processing
movie_data=pd.read_csv('Movie_Genre.csv')
movie_count=movie_data.groupby('genres')['movieId'].count()
movie_count=movie_count.reset_index()

# Create list of movies
movie=movie_data['title']
movie=list(movie)

# Classify each movie into its genre and form a genre table from data, columns - Genre name; Rows - movie_ID"""
d=[]
# movie_count stores the types of genre and no of ID of each genre in data
for i in range(0,len(movie_count)):
    # making column of the genre and add all the movieID     
    d.append(movie_data['movieId'][movie_data['genres'] == movie_count['genres'][i]])
    d[i]=pd.DataFrame(d[i])
    d[i]=d[i].reset_index()
    
for i in range(0,11):    
    d[i]=d[i].rename(columns={'movieId':movie_count['genres'][i]})    

data=d[0]
for i in range(1,len(d)):
    data=pd.concat([data,d[i]],join='outer',axis=1)
    
data=data.drop(columns=['index'])
genre=data.columns
genre_data=data

# movie ID action space from where we recommend """
movie_Id=movie_data['movieId']
movie_Id=list(movie_Id)

###############################################
# RL has State Space of each genre"""
State_Space_class1=[]
State_Space_class2=[]
State_Space_class3=[]
State_Space_class4=[]
State_Space_class5=[]
State_Space_class6=[]
State_Space_class7=[]
State_Space_class8=[]
State_Space_class9=[]
State_Space_class10=[]
State_Space_class11=[]


# Initialize the Q table for each genre with zero
class1_Table=np.zeros([0])
class2_Table=np.zeros([0])
class3_Table=np.zeros([0])
class4_Table=np.zeros([0])
class5_Table=np.zeros([0])
class6_Table=np.zeros([0])
class7_Table=np.zeros([0])
class8_Table=np.zeros([0])
class9_Table=np.zeros([0])
class10_Table=np.zeros([0])
class11_Table=np.zeros([0])

# Dictionary for calling required table and state_space and mapping
Dict = {'State_Space_1': State_Space_class1,
        'State_Space_2': State_Space_class2,
        'State_Space_3': State_Space_class3,
        'State_Space_4': State_Space_class4,
        'State_Space_5': State_Space_class5,
        'State_Space_6': State_Space_class6,
        'State_Space_7': State_Space_class7,
        'State_Space_8': State_Space_class8,
        'State_Space_9': State_Space_class9,
        'State_Space_10': State_Space_class10,
        'State_Space_11': State_Space_class11,
        'class1':class1_Table,
        'class2':class2_Table,
        'class3':class3_Table,
        'class4':class4_Table,
        'class5':class5_Table,
        'class6':class6_Table,
        'class7':class7_Table,
        'class8':class8_Table,
        'class9':class9_Table,
        'class10':class10_Table,
        'class11':class11_Table}

# Dictionary for calling required table,state_space, Genre and mapping
Dict2={'State_Space':['State_Space_1','State_Space_2','State_Space_3','State_Space_4','State_Space_5','State_Space_6','State_Space_7','State_Space_8','State_Space_9','State_Space_10','State_Space_11'],
       'Table':['class1','class2','class3','class4','class5','class6','class7','class8','class9','class10','class11'],
       'Genre':[movie_count['genres']]}

# Creating State_Space
for i in range(0,len(movie_count)):
    Dict[Dict2['State_Space'][i]].append(list(it.combinations(movie_data['movieId'][movie_data['genres'] == movie_count['genres'][i]],1)))
    Dict[Dict2['State_Space'][i]].append(list(it.combinations(movie_data['movieId'][movie_data['genres'] == movie_count['genres'][i]],2)))
    
# Creating and initialize Q table of each genre"""
for i in range(0,len(movie_count)):    
    Dict[Dict2['Table'][i]]=np.zeros([len(Dict[Dict2['State_Space'][i]][0])+len(Dict[Dict2['State_Space'][i]][1]),len(movie)])  
      
# let S be a state of user, Finding index of the state of the user
Index=[]
def INITIAL_STATE_INDEX(S,State_Space_name):
    """
    Parameters:
    S (tuple): State - previous history [movies]
    State_Space_name - as we have divided according to genre, so goes the state space of that genre
    Returns:
    Index:Index of that state in State_Space/table row
    """
    Secondary_State=[]
    n=len(S)
    State_Space=Dict[State_Space_name]
    #Storing variation of S as rearrangement matters
    Secondary_State=tuple(list(it.permutations(S, n)))
    # As our state space contains 1,2 combinations of the ID, in seperate list so that is decided by n to which list to search 
    if n==1:
        for i in range(0,len(Secondary_State)):
            if Secondary_State[i] in State_Space[0]:
                Index.append(State_Space[0].index(Secondary_State[i]))
                break
            # just to avoid error if item is not present I have put esle statement, But that condition will not occur
            else :
                Index.append(0)
        return(Index[0])

    else:
        for i in range(0,len(Secondary_State)):
            if Secondary_State[i] in State_Space[1]:
                Index.append(State_Space[1].index(Secondary_State[i])+len(State_Space[0]))
                break
            # just to avoid error if item is not present I have put esle statement, But that condition will not occur     
            else:
                Index.append(0)
        return(Index[-1])

# Currently S contaions only previous history of user
# New We need to find the genre of each history like - its [genre, table, state_space, its index in state space]
# Preprocess function to perform above operation
       
table=[]
table3=[]
def PREPROCESS(S):
    """
    Parameters:
    S (tuple): State - previous history [movies]
    Returns:
    table: table that contains [movies, genre,  state_space, table, its row in that table]
    """
    # movie genre[genre of movie, movie name] in seperate rows for each genre type
    movies_genre=[]
    # genre_data reads all class and there titles from  csv file
    for i in range(0,len(S)):
        for col in genre_data.columns:
            if S[i] in list(genre_data[col]):
                movies_genre.append((col,S[i]))
    # Table contaion [movie1,....movie_i, class of movies in it]
    # Now from movie genre, look for the table, we combine the id of same genre together and different genre in different row of tabe     
    f=0
    for i in range(0,len(movies_genre)):
        r=[]
        c=movies_genre[i][0]
        for j in range(0,len(movies_genre)):

            if movies_genre[j][0]==c:
                r.append(movies_genre[j][1])
        r.append(c)
        r=tuple(r)
        table3.append(r)
        f = 0
        for k in range(0,len(table)):
            if table[k][-1]==table3[-1][-1]:
                f=1
        if f==0:
            table.append(r)
    table3.clear()
    # now table contains [movies,class of movies]
    # Then we find its statespace, table and row
    for j in range(0,len(table)):
        for i in range(0,len(Dict2['Genre'][0])):
            if table[j][-1] == Dict2['Genre'][0][i]:
                table[j]=list(table[j])
                table[j].append(Dict2['State_Space'][i])
                table[j].append(Dict2['Table'][i])

    # now if all the movie is of same genre len(S) will be three, but our state space does not have 3 combinations of it
    # so we break it into 3C2 combinations and search for it in state space          
    if len(table)==1 and len(table[0][0:-3])==3:
        t=table.copy()
        table.clear()
        Secondary_State2=[]
        S1=t[0][0:-3]
        z=[]
        Secondary_State2=tuple(list(it.combinations(S1, 2)))
        for i in range(0,3):
            s=Secondary_State2[i]
            z.extend([s[0],s[1],t[0][3],t[0][4],t[0][5]])
            table.append(z[-5:])
        z.clear()
    state=[]
    # after that we find the row of that state in Q table for Q update and taking decision for recommendation
    for i in range(0,len(table)):
        state.append(INITIAL_STATE_INDEX(table[i][0:-3],table[i][-2]))
        table[i].extend([state[-1]])
        state.remove(state[-1])
        Index.remove(Index[-1])    
    return table
# table returned from it will be of form - [[movie_Ids,genre,State_Space_name,Q_Table-Name,row of Q table containing that state],     [movie_Ids,genre,State_Space_name,Q_Table-Name,row of Q table containing that state]] 
# now if the user clicks the movie it means he/she has wathced the movie and hence user state will change
# NEW_STATE changes the user state according to the click of  user

# NEW_STATE  function  is used to find the next state after action is taken
def NEW_STATE(S,action_idx):
    New_State=(S[1],S[2],action_idx)
    return (New_State)

# find the index of present state
present_ind=[]
def PRESENT_INDEX(S):
    for i in range(0,len(S)):
        present_ind.append(S[i])
    return(present_ind)

# For next recommendation of the same state, as a single state many have 2-3 substate so, for 1st state if some recommendation
# is given, that is stored in Rec_List, so in next sub state we have to remove those from recommendation list otherwise there may be repeated recommendation
rec_ind=[]
def REC_INDEX(S):
    for i in range(0,len(S)):
        rec_ind.append(S[i])
    return(rec_ind)

# Now to capture and update Q values
# if user clicks on the recommended movie then we assign reward
# Q values of the clicked movies are updated accordingly in their respective Q TABLE_UPDATE
# UPDATE_CLICK_Q_VALUE it updates the Q valus of the clicked movies
def UPDATE_CLICK_Q_VALUE(reward,Click):
    """
    Parameters:
    Reward(Integer): Reward value, This function is called only after user clicks the recommendation and reward is given to update the Q value
    Click(List): We provide list of recommendation, Click list contains the index of movies clicled by user from recommendation list
    Returns:
    NULL:Update Q values in respective table and update the state of the user
    """
    global table
    global S1
    S1=list(S1)
    S=S1.copy()
    for i in range(0,len(Click)):
        # as we provide recommendation according to substate, maximum of 3 recommendation is provided, so if user clcik in any of it, we need to know from which substate it was recommended to update the state Q_value accordingly
        # K is that value, Click stores the index of the click item in the Rec_List from that we decide to which substate that recommendation belongs to       
        n=Click[i]
        if n==0:
            k=0
        elif n==1:
            k=1
        else:
            if len(table2)==3:
                k=2
            else:
                k=len(table2)-1
        # Now we find the row and column where we have to update the Q value            
        #those values are stored in the table formed by PREPROCESS function        
        column=Action[Click[i]]-1
        row=table2[k][-1]
        old_value = Dict[table2[k][-2]][row,column]
        movie_name=Action[Click[i]]
        # after user action we have to form the new state, new state max value is used to update the Q value
        # as this algorithm looks for the future reward, so we need to find the new state      
        new_S=NEW_STATE(S1,Action[Click[i]])
        # from new state, we have substate in a State, so we need to see our substate has gone to which new substate it belongs
        # we take the max Q value of that next substate form Q update, that next sub state is found by calling the PREPROCES function for new State formed      
        PREPROCESS(new_S)
        for j in range(0,len(table)):
            if movie_name in table[j][0:-4]:
                next_state_index=table[j][-1]
                next_max = np.max(Dict[table[j][-2]][next_state_index])
        new_value = (1 - alpha) * old_value + alpha *(reward + gamma * (next_max-old_value))
        Dict[table2[k][-2]][row,column] = new_value
        # after all the updates we update the state of the user  
        S1=new_S
        table.clear()
        
        
# UPDATE_NOT_CLICK_Q_VALUE it updates the Q valus of the Not_clicked movies"""
def UPDATE_NOT_CLICK_Q_VALUE(reward1,Not_Click):
    """
    Parameters:
    Reward: Negative reward, as recommendation is ignored, Updates the Q value
    Not_Click(List) - We provide list of recommendation, Not_Click list contains the index of movies not clicled by user from recommendation list
    Returns:
    NULL: Update Q values in the respective tables
    """
    reward= -reward1
    # K value same as explaind in the above function
    for i in range(0,len(Not_Click)):
        n=Not_Click[i]
        if n==0:
            k=0
        elif n==1:
            k=1
        else:
            if len(table2)==3:
                k=2
            else:
                k=len(table2)-1
        # Update of the Q value    
        column=Action[Not_Click[i]]-1
        row=table2[k][-1]
        old_value = Dict[table2[k][-2]][row,column]
        next_max=old_value
        new_value = (1 - alpha) * old_value + alpha *(reward + gamma * (next_max-old_value))
        Dict[table2[k][-2]][row,column]=new_value
        table.clear()

# Hyperparameters
alpha = optim.alpha
gamma = optim.gamma
epsilon = optim.epsilon
Action=[]
Rec_List=[]
# Recommendation according to states takes input as table[i]
def RECOMMENDATION(List_I):
    """
    Parameters:
    table[i]:table formed by PREPROCESSED function, stores Substate, its genre, its table and row of the sub_state in that table
    Returns:
    Recommendation: Recommendation for that Sub_State
    """
    global present_ind
    global S1
    S=List_I[0:-4]
 
    S_ind=PRESENT_INDEX(List_I)
    S_ind=set(S_ind)
    day=set(movie_Id)
    if len(Rec_List)!=0:
        # removes the item from recommendation list if previously recommended in substate of same state    
        Rec_List_ind=REC_INDEX(Rec_List)
        Rec_List_ind=set(Rec_List_ind)
        rec1=day-S_ind
        rec=rec1-Rec_List_ind
        Rec_List_ind=list(Rec_List_ind)
    else:
        rec=day-S_ind
    rec=list(rec)
    S_ind=list(S_ind)
    present_ind.clear()
    rec_ind.clear()
      
    state = List_I[-1] #row of Qtable as stored in table 
    # check whether to explore or exploit
    #explore
    if random.uniform(0, 1) <= epsilon or np.argmax(Dict[List_I[-2]][state])<= epsilon:
    # Check the action space
        action1=random.choice(rec)
        Action.append(action1)

    #exploit
    else:
    # Check the learned values
        aa= np.array(Dict[List_I[-2]][state])
        idx=aa.argsort()
        idx=list(idx)
        for i in range(0,len(S_ind)):
            idx.remove(movie_Id.index(S_ind[i]))
        if len(Rec_List)!=0:
            Rec_List_ind=[]
            for i in range(0,len(Rec_List)):
                Rec_List_ind.append(movie_Id.index(Rec_List[i]))
                if Rec_List_ind[-1] in idx:
                    idx.remove(Rec_List_ind[i])
        Action.append(idx[-1]+1)
        
    Index.clear()
    rec_ind.clear()
    return(Action[-1])

# A single user can be in multiple states a/c to Q_tables so we stack all the recommendation corresponding
#each state and create the recommendation List
Recommendation_List=[]    
def RECOMMENDATION_LIST(S):
    """
    Parameters:
    S: State - previous 3 history
    Returns:
    Recommendation_List: Recommendation from all sub state
    """
    PREPROCESS(S)
    for i in range(0,len(table)):
        a=RECOMMENDATION(table[i])
        if a not in Rec_List:
            Rec_List.append(a)
    for i in range(0,len(Action)):
        Recommendation_List.append(movie[movie_Id.index(Action[i])]) 
    return Recommendation_List


# now after user action we need to update the Tables accordingly
# TABLE_UPDATE function updates all the related tables update the state of the user
# gives back new recommendation list according to the new states
# takes input as list Click - which store the index of item from Rec_List which was clicked by user
# updates all the q values and form new state and again provide the list of recommendation according to new state
Click=[]
Not_Click=[]
def TABLE_UPDATE(Click):
    global Action
    global table
    global table2
    global Not_Click
    global S1
    Click2=[]
    # as only list of clicked items are given, we need to update not_clicked items also, so we create that    
    for i in range(0,len(Click)):
        Click2.append(Action[Click[i]])
    Click2=set(Click2)
    Action=set(Action)
    Not_Click1=Action- Click2
    Action=list(Action)
    Not_Click1=list(Not_Click1)
    for i in range(0,len(Not_Click1)):
        if Not_Click1[i] in Action:
            Not_Click.append(Action.index(Not_Click1[i]))
    #  Not_Clicked item is created
    table2=table.copy()
    table.clear()
    if len(Click)!=0:
        UPDATE_CLICK_Q_VALUE(15,Click)
        
    if len(Not_Click)!=0:   
        UPDATE_NOT_CLICK_Q_VALUE(5,Not_Click)
        
    Action.clear()
    table2.clear()
    Index.clear()
    Rec_List.clear()
    Not_Click.clear()
    Not_Click1.clear()
    Recommendation_List.clear()
    Click.clear()
    

# Now all updates are done and according to new state of the user, we provide a new list of recommendation to it  

# How to run the algorithm
# Just call RECOMMENDATION_LIST(S) function, input state of the user   
# after user action, click or ignore
# TABLE_UPDATE(Click)function input- Click list - contains index of items clicked by the user from the recommendation list provided.   
# State contains the movie_Id of the previous 3 movies clicked by user
# State is not according to user, but every user to which we have to recommend
# will fall in any of the state which we have formed.
# for each user, every time he/she comes we need to see his previous 3 history and form state and input in according as written in above line
# even if user has one history, it will be able to recommed a better product

# Evaluation
# for  training, we need to make bit of changes in the above code
# Data for training- State(1,2,3) id of the movie watched as state
# item recommended in that state
# Data columns-State, Item recommended in that state

test_set=pd.read_csv('State_Data.csv')
# Count of the state data each state has
State_count=test_set.groupby('State')['recommendation'].count()
State_count=State_count.reset_index()
# State
State=test_set.iloc[:,2].values
State=list(State)
# clicked item in thatstate
Recommended=test_set.iloc[:,3].values
Recommended=list(Recommended)

# call the RECOMMENDATION_LIST(S) and see the item clicked in it,
# now if that clicked item is in the list, we append its index in the Click list
# Call TABLE_UPDATE(Click) function to update Q value
# How to update -- 
# we input the State[i] and take recommendation from algorithm,
# We check the rec_list, if it contains the clicked itmes, its fine
# if it doesnot, then we append that item in the clicked list and update click as well as not clicked items

for i in range(0,len(State)):
    S1=[]
    a=State[i]
    b=int(a%1000)
    c=int((a/1000)%1000)
    d=int((a/1000000))
    S1.extend([d,c,b])
    RECOMMENDATION_LIST(S1)
    Action.clear()
    Action.extend(Rec_List)
    if Recommended[i] not in Action:
       Action.append(Recommended[i]) 
    Click.append(Action.index(Recommended[i]))
    if i%1000==0:
        print("--i--",i)    
    TABLE_UPDATE(Click) 

#Plot to see the optimization of states Q value, consider any one state of any one table   

   









               
