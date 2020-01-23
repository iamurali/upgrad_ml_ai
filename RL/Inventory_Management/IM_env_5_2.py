from gym import spaces
import numpy as np


m=50    	#max capacity of warehouse
K=3      	#constant part of order cost (K in document), can be cost of fuel
c=4      	#variable part of order cost (c(a_t) in document)
h=0.0025    #holding cost 
p=4.5      	#selling price of product is such that PROFIT = 12.5%
R=K		 	#return cost = K because cost of fuel is same for to and fro journeys
lamda_mon=16    #lambda for poisson distribution
lamda_tue=31
lamda_wed=15
lamda_thu=32
lamda_fri=30
lamda_sat=8
lamda_sun=42

day_mapping = {0:'mon',1:'tue',2:'wed',3:'thu',4:'fri',5:'sat',6:'sun'}

class InventoryEnv():

    def __init__(self):
        self.action_space = spaces.Discrete(m+1)
        self.inventory = np.random.choice(np.arange(0,m+1))       
        self.day = np.random.choice((0,1,2,3,4,5,6))
        self.state = (self.inventory,self.day)

        # Start the first round
        self.reset()

    def demand(self, day):
    	if day == 0:
    		return np.random.poisson(lamda_mon)
    	elif day == 1:
    		return np.random.poisson(lamda_tue)
    	elif day == 2:
    		return np.random.poisson(lamda_wed)
    	elif day == 3:
    		return np.random.poisson(lamda_thu)
    	elif day == 4:
    		return np.random.poisson(lamda_fri)
    	elif day == 5:
    		return np.random.poisson(lamda_sat)  
    	else:
    		return np.random.poisson(lamda_sun)
        

    def transition(self, x_t_1, a_t_1, d_t):

        if x_t_1[1] <6:
        	next_day = x_t_1[1]+1
        else:
        	next_day = 0    

        stock_after_sales = max(x_t_1[0] - d_t, 0)	#first this is calculated because this cannot go below 0
        stock_EOD = min(stock_after_sales + a_t_1,m)	#this is calculated second because this is added after the demand has been satisfied
        
        #note that state includes the order which was just delivered
        return (stock_EOD, next_day) 

    def reward(self, x_t_1, a_t_1, d_t):
        # x_t_1 = state(today-1), x_t = state(today)    #x_t_1 is the first element of the state tuple
        #Similarly for a = action and d = demand


        #1. EXPECTED INCOME
        expected_income = p * min(d_t,x_t_1)      #quantity sold=d,i.e.,demand. But if d>x, then quantity sold=x,i.e.,stock from last night
        
        #2. ORDER COST
        fixed_order_cost = K * (a_t_1 > 0)
        variable_order_cost = c
        order_cost = fixed_order_cost + variable_order_cost * a_t_1
        
        #3. HOLDING COST
        holding_cost = h * x_t_1
                
        #4. OPPORTUNITY COST
        actual_demand = d_t
        demand_satisfied = x_t_1 #because in d>x, we can only sell x
        profit = p - c
        #profit = 0.5
        opportunity_cost = profit * (actual_demand - demand_satisfied) * (actual_demand>demand_satisfied)
               
        #5. RETURN COST
        stock_after_sales = x_t_1 - d_t
        stock_arrived = a_t_1
        return_cost = R * (stock_after_sales + stock_arrived > m)    #can't use x_t_1 directly because it will be cut off at m

        #6. MONEY BACK
        stock_after_sales = x_t_1 - d_t
        stock_arrived = a_t_1
        money_back = c * (stock_after_sales + stock_arrived - m)  * (stock_after_sales + stock_arrived > m)

        r = expected_income - order_cost - holding_cost - opportunity_cost - return_cost + money_back
        return r

    def initial_step(self, state, action):
        assert self.action_space.contains(action)     #to check that action is a discrete value less than m
        obs = state

        if state[1]<6:
            demand = self.demand(state[1]+1)    
        else:
            demand = self.demand(0)        

        obs2 = self.transition(obs, action, demand)       #next_state

        return obs2



    def step(self, x_t_1, a_t_1):   
        assert self.action_space.contains(a_t_1)     #to check that action is a discrete value less than m
        obs = x_t_1             #at the beginning, state is picked up from the contructor. 


        if x_t_1[1]<6:
            d_t = self.demand(x_t_1[1]+1)    
        else:
            d_t = self.demand(0)

        obs2 = self.transition(obs, a_t_1, d_t)       #next_state


        reward = self.reward(x_t_1[0], a_t_1,  d_t)

        return obs2, reward



    def reset(self):
        return self.state
