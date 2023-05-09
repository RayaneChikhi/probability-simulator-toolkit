#Probability simulator
from math import *
from matplotlib import pyplot as plt
from random import *
import numpy as np
from turtle import *
from scipy.interpolate import lagrange
from numpy.polynomial.polynomial import Polynomial
import scipy.stats as sp
from collections import Counter
import networkx as nx
import scipy.special as spp
from scipy.integrate import quad
import scipy

#User made law:

class discrete_law:
    def __init__(self, state, parameter, weights):
        self.parameter = parameter
        self.weights = weights
        self.state = state
    
    def law(self, v=1):

        return (self.state, choices(self.state, weights = self.weights, k=v), self.weights)

    



#Returns average of a list
def avg(l):
    return sum(l)/len(l)

#Returns expected value of a law:
def expectedValue(law, parameter):
    match law.__name__ :
        case 'uniform':
            return (parameter+1)/2
        case 'bernoulli':
            return parameter
        case 'binomial':
            n,p = parameter
            return n*p
        case 'poisson':
            return parameter
        case 'lognormal':
            sigma,mu = parameter
            return exp(mu + ((sigma)**2)/2)
        case 'rademacher':
            return 0
        case 'zeta_law':
            if parameter<=2:
                return float("inf")
            else:
                s=parameter
                return spp.zeta(s-1)/spp.zeta(s)
def variance(law,parameter):
    match law.__name__ :
        case 'uniform':
            return (parameter**2-1)/12
        case 'bernoulli':
            return parameter*(1-parameter)
        case 'binomial':
            n,p = parameter
            return n*p*(1-p)
        case 'poisson':
            return parameter
        case 'lognormal':
            sigma,mu = parameter
            return (exp(sigma**2 - 1)*exp(2*mu+sigma**2))
        case 'rademacher':
            return 1
        case 'zeta_law':
            if parameter<=3:
                return float('inf')
            else:
                s = parameter
                return (spp.zeta(s)*spp.zeta(s-2)-(spp.zeta(s-1))**2)/(spp.zeta(s)**2)

#Generates number (100 by default) numbers randomly using a certain law and a certain parameter associated to said law. 
#To illustrate the law of large numbers, we also sum all the values obtained and normalize them. According to the law of large numbers,
#we should obtain a graph centered around the expected value of our random variable.
def generate(law, parameter,number=100,showNormalized=False, q=100):
    
    ev = expectedValue(law,parameter)
    if showNormalized:
        figure, axis = plt.subplots(1,3)
    else:
        figure,axis = plt.subplots(1,2)

    state, values, weights = law(parameter,number)
    if showNormalized:
        u = np.array(values)
        
        for i in range(q-1):
            state,vars,weights = law(parameter,number)
            u = u + np.array(vars)
        u = u/q

    n=len(state)
    print("Expected value:", ev)
    print("Simulated average:", avg(values))
    if showNormalized:
        o=max([abs(x-ev) for x in u])
        axis[2].plot([i for i in range(number)], u)
        axis[2].plot([i for i in range(number)],[ev for i in range(number)], color='r')
        axis[2].plot([i for i in range(number)], [ev+o for i in range(number)], color='g', linestyle='dashed')
        axis[2].plot([i for i in range(number)],[ev - o for i in range(number)], color='g', linestyle='dashed')
        axis[2].set_xlabel('k-th try')
        axis[2].set_ylabel('(X₁ + ... + Xₚ)/p')

        axis[2].legend(loc="upper left")
        axis[2].legend(['Normalized sums for each try k', 'Expected Value'])
        axis[2].annotate('Highest deviation to expected value:' +  str(o),(0,0),(0,-30),xycoords='axes fraction', textcoords='offset points', va='top')
    axis[1].hist(values,bins=15, facecolor = '#2ab0ff', edgecolor='#169acf', linewidth=0.5)
    axis[1].set_xlabel('k')
    axis[0].bar(state, weights, facecolor = '#2ab0ff', edgecolor = '#169acf', linewidth=1)
    axis[1].set_ylabel('Number of times the value k is chosen')
    axis[0].set_ylabel('Probability of k being chosen')
    axis[0].set_xlabel('k')
    
    plt.show()

#For continuous laws:

def gen_continuous(law, parameter,v):
    density, values, k = law(parameter,v)
    ev= expectedValue(law,parameter)
    v = variance(law,parameter)
    print("Expected value:", ev)
    print("Variance:", v)
    count, bins, ig = plt.hist(values, 100, density=True, align='mid')
    x = np.linspace(min(bins), max(bins), 10000)
    plt.plot(x, [density(y) for y in x])
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.axis('tight')
    plt.show()



################################ LAWS

#Generates v numbers chosen uniformly in [|1,n|]
def uniform(n,v=1):
    variable=[]
    for i in range(v):
        variable.append(randint(1,n))
    return ([y for y in range(1,n+1)], variable,[1/n for i in range(1,n+1)])

#Rademacher law 
def rademacher(p=1,v=1):
    return ([-1,1], choices([-1,1], weights=(1/2,1/2),k=v), [1/2, 1/2])
#Generates v numbers using a bernoulli law of parameter p
def bernoulli(p,v=1):
    if not(0<=p and p<=1):
        print("0<=p<=1 expected")
        return None
    return ([0,1], choices([0,1], weights=(1-p,p),k=v), [1-p, p])

#Zeta law of parameter s

def zeta_law(s,v = 1):
    if s<=1:
        print("s>1 expected")
        return None
    u = ceil(30)
    a = [i for i in range(1,u+1)]
    b = [1/((k**s)*spp.zeta(s)) for k in a]
    return (a, choices(a, weights=b, k = v), b)

#Generates a binomial random variable of parameters (n,p)
def binomial(tuple_parameter,v=1):
    n,p = tuple_parameter
    a=[i for i in range(n+1)]
    b=[comb(n,k)*(p**k)*((1-p)**(n-k)) for k in a]
    if not(0<=p and p<=1):
        print("0<=p<=1 expected")
        return None
    return (a, choices(a, weights=b,k=v),b)

#Generates a Poisson distribution of parameter Lambda for every values taken by the variable (up to 4*lambda by default where the probabilities are really small)
def poisson(Lambda,v=1):
    a=[i for i in range(4*ceil(Lambda)+1)]
    u=[exp(-Lambda)*(Lambda**k)/(factorial(k)) for k in a]
    return (a, choices(a, weights = u,k=v), u)

def lognormal(tuple_parameter, v=100):
    sigma,mu = tuple_parameter
    s = np.random.lognormal(mu, sigma, v)

    def f(x):
        return (1/(x*sigma*sqrt(2*pi)))*exp(-((log(x)-mu)**2)/(2*sigma**2))
    return (f, s, v)


########################################################################################################################################
#Generates an n by m matrix where the entries of said matrix are values taken by a random variable of law "law" and parameter "parameter"
def gen_matrix(n,m, law, parameter):
    matrix = []
    for i in range(n):
        state,variable,weight = law(parameter,m)
        matrix.append(variable)
    return matrix




#RANDOM WALKS
def randwalk_1D(law=[1/3,1/3,1/3]):
    step_n = 100000
    step_set = [[-1], [0], [1]]
    origin = np.zeros((1,d))
    d=1

    
    steps = choices(step_set, weights = law, k = step_n * d)
    steps = np.array(steps)
    path = np.concatenate([origin, steps]).cumsum(0)
    start = path[:1]
    stop = path[-1:]

    fig = plt.figure(figsize=(8,4),dpi=200)
    ax = fig.add_subplot(111)
    ax.scatter(np.arange(step_n+1), path, c='blue',alpha=0.25,s=0.05)
    

    
    plt.title('One dimensional random walk')
    plt.show()


def randwalk_2D():
    dims = 2
    step_n = 10000
    step_set = [-1, 0, 1]
    origin = np.zeros((1,dims))

    step_shape = (step_n,dims)
    steps = np.random.choice(a=step_set, size=step_shape)
    path = np.concatenate([origin, steps]).cumsum(0)
    start = path[:1]
    stop = path[-1:]

    fig = plt.figure(figsize=(8,8),dpi=200)
    ax = fig.add_subplot(111)
    ax.scatter(path[:,0], path[:,1],c='blue',alpha=0.25,s=0.05)
    ax.plot(path[:,0], path[:,1],c='blue',alpha=0.5,lw=0.25,ls='-');
    
    plt.title('2D Random Walk')
    plt.show()

#Missing 3D
############################################################

#Around Weierstrass' theorem:

def bernstein(f,x,n):
    val = 0
    for k in range(n+1):
        val+= (comb(n,k))*(x**k)*((1-x)**(n-k))*f(k/n)
    return val

def weierstrass(f,n):
    values = np.array([bernstein(f,1/k,n) for k in range(1,n+1)])
    x = np.array([1/k for k in range(1,n+1)])
    poly = lagrange(x,values)
    return Polynomial(poly.coef[::-1])

def graph_weierstrass(f,n):
    x = np.linspace(0,1)
    fig, ax = plt.subplots(1,1)
    y = [bernstein(f,k,n+1) for k in x]
    y2=[f(k) for k in x]
    plt.plot(x,y2,linestyle='dotted',linewidth='3', label = 'f(x)')
    plt.plot(x,y, label = 'Bn(f)(x)')
    ax.set_xlabel('x')
    ax.legend(loc='upper left')
    plt.figure().set_figwidth(5)
    plt.text(0.5,0.5, 'Bn(f)(x) = ' + str(weierstrass(f,n)), horizontalalignment='center', verticalalignment='center')
    
    print(str(weierstrass(f,n)))
    plt.axis('off')

    plt.show()

######################################### Around permutations
def fisheryates(n):
    s = [i for i in range(1,n+1)]
    a=[i for i in range(1,n+1)]
    for i in range(n):
        j = randrange(i,n)
        s[j],s[i]=s[i],s[j]
    return [a,s]

def num_fixedpoints(n,tries):
    ans = [0 for i in range(n+1)]

    for i in range(tries):
        s = fisheryates(n)[1]
        s_fixedpoints=0
        for j in range(n):
            if s[j]==j+1:
                s_fixedpoints+=1
        ans.append(s_fixedpoints)
    plt.hist(ans,bins=n,facecolor = '#2ab0ff', edgecolor='#169acf', linewidth=0.5)
    state, variable, weight = poisson(1, tries)
    plt.hist(variable, bins=n,facecolor = 'red', edgecolor='#169acf', linewidth=0.5)
    plt.legend(['Number of times point is fixed','Poisson 1 distribution'])
    plt.show()


def cycle_lengths(n):
    s = fisheryates(n)[1]
    
    cycles = [0 for i in range(n)]
    visited = []
    for i in range(1, n+1):
        if i not in visited:
            cycle = []
            j = i
            while j not in visited:
                visited.append(j)
                cycle.append(j)
                j = s[j-1]
            cycles[len(cycle)-1]+=1
    
    return cycles

def cycle_structure_theorem(n,m):
    variable=np.array([0 for i in range(n)])
    for i in range(m):
        variable+=np.array(cycle_lengths(n))

    alpha = 0.5

    fig, ax = plt.subplots(1,2)
    ax[0].plot([spp.gamma(alpha)/alpha * 1/(i*(i+1)) for i in range(1, n+1)],[i for i in range(1,n+1)])
    ax[1].bar([i for i in range(1,n+1)], variable)

    plt.show()



############# Around Wigner's semi-circle law for Gaussian matrices #################
def geneigen(law, parameter, N = 10):
    A = gen_matrix(N,N, law, parameter)
    ev=expectedValue(law,parameter)
    A=np.array([np.array(u) - ev for u in A])
    B = 2**-0.5 * (A + A.T)
    D = np.linalg.eigvals(B)  
    return np.real(D)


def random_matrix_eigenvals(law, parameter,n_matrices = 10000, N = 9):
    eig = np.zeros([n_matrices, N]) 
    for i in range(n_matrices):
        eig[i, :] = geneigen(law, parameter, N) 
    eig = eig / np.sqrt(N)
    spacing = np.diff(np.sort(eig)) 
    spacing = spacing / np.mean(spacing) 
    
    return eig, spacing

def plot_random_matrix_eigenvalues(law, parameter, n_matrices = 100000, N = 10):
    eig, spacing = random_matrix_eigenvals(law, parameter, n_matrices, N)
    plt.subplot(1, 2, 1)
    plt.title("Wigner semicircle law")
    plt.hist(eig.ravel(), 100, density = True)
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.title("Random Matrix Eigenvalue Spacing")
    plt.hist(spacing.ravel(), 100, density = True)
    plt.grid(True)
    plt.legend(["Wigner Surmise", "Eigenvalue Spacings"])
    plt.show()
###############################################################

########## Around the probability of two numbers being mutually prime

def coprime(n,m):
    return gcd(n,m)==1

def sim_coprime(N, tries):
    ans=0
    for i in range(tries):
        n = randint(1,N)
        m = randint(1,N)
        if coprime(n,m):
            ans+=1
    return ans/N

################################################################### Around the central limit theorem

def clt(law, parameter, n,m):
    e = expectedValue(law,parameter)
    v = variance(law, parameter)
    l=[]
    for k in range(1,m+1):
        state, var, weights = law(parameter, n)
        X=sum(var)
        X = (X - n*e)/(n*sqrt(v))
        l.append(X)
    
    plt.hist(l,facecolor = '#2ab0ff', edgecolor='#169acf', linewidth=0.5)
    plt.show()

#################################################################### Birthday problem
def commonBirthday(L,i,seen):
    for j in range(len(L)):
        if L[i]==L[j] and j not in seen:
            seen.append(j)
            return True
    return False

def birthday(n,N=1000):
    results=[]
    for i in range(1,n+1):
        to_average=[]
        for j in range(N):
            state, birthdays, weights = uniform(365,i)
            numberOfCommonBirthdays=0
            counts = dict(Counter(birthdays))
            duplicates = {key:value for key, value in counts.items() if value > 1}
            for key in duplicates:
                numberOfCommonBirthdays+=duplicates[key]
            to_average.append(numberOfCommonBirthdays)
        results.append(avg(to_average))
    fig, ax = plt.subplots(1,2)
    ax[0].title.set_text('Number of common birthdays for each classroom')
    ax[0].plot([i for i in range(1,n+1)], results)
    ax[0].set_xlabel('Classroom size')
    ax[0].set_ylabel('Number of common birthdays')
    maxValue=max(results)
    ax[1].title.set_text('Probability distribution in theory')
    ax[1].set_xlabel('Classroom size')
    ax[1].set_ylabel('Probability of two people sharing the same birth date')
    ax[1].bar([i for i in range(1,n+1)],[(1 - (factorial(365) / (factorial(365-i) * 365**i))) for i in range(1,n+1)])
    plt.show()


#################################################################
##Regressions:

def linear_regression(x, y):     
    N = len(x)
    x_mean = avg(x)
    y_mean = avg(y)
    x=np.array(x)
    y=np.array(y)
    B1_num = ((x - x_mean) * (y - y_mean)).sum()
    B1_den = ((x - x_mean)**2).sum()
    B1 = B1_num / B1_den    
    num = (N * (x*y).sum()) - (x.sum() * y.sum())
    den = np.sqrt((N * (x**2).sum() - x.sum()**2) * (N * (y**2).sum() - y.sum()**2))
    R = num / den
    B0 = y_mean - (B1*x_mean)
    reg = 'y = {} + {}x'.format(B0, round(B1, 3))
    
    return (R, B0,B1,reg)

def exp_regression(x,y):
    #Of the form y = C*exp(alpha*x) ie ln(y) = ln(C) + alpha*x 
    x=np.array(x)
    y=np.array(y)
    lny = np.log(y)
    r,B0,B1,reg = linear_regression(x,lny)
    C=exp(B0)
    alpha=B1
    return (r, C, alpha, 'y =' + str(C) + 'exp(' + str(alpha) + 'x)')

################################################## Around random graphs

def generate_graph(adjacency,weights=None):
    G = nx.DiGraph(Directed=True)
    nodes = adjacency.keys()
    G.add_nodes_from(nodes)

    for u in nodes:
        neighbours = adjacency[u]
        for j in range(len(neighbours)):
            if weights!=None:
                G.add_edge(u, neighbours[j], weight=weights[(u,neighbours[j])], length=30)
            else:
                G.add_edge(nodes[u],neighbours[j], length = 1)
    pos = nx.spring_layout(G, seed=7)
    nx.draw_networkx_nodes(G, pos, node_size=700,node_color='yellow')

    # edges
    nx.draw_networkx_edges(G, pos, width=5)
    nx.draw_networkx_edges(
        G, pos, width=2, alpha=1, edge_color="grey", style="dashed"
    )

    # node labels
    nx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif",font_color='orange')
    # edge weight labels
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels,font_size=10)

    ax = plt.gca()
    ax.margins(0.08)
    plt.axis("off")
    plt.show()
    return G
    
def random_weighted_graph(adjacency, law, parameter):
    edges = 0
    
    for i in range(len(adjacency)):
        edges+=len(adjacency[i][1])
    m = gen_matrix(len(adjacency), edges, law, parameter)
    weights = dict()
    for i in range(len(adjacency)):
        for j in range(len(adjacency[i][1])):
            weights[(adjacency[i][0],adjacency[i][1][j])] = m[i][j]
    return weights

adjac = [[1, [3,2]],[2,[1,3]],[5,[4,3]],[4,[3,2,1]]]
#w=random_weighted_graph(adjac,lognormal,(2,0.1))
#generate_graph(adjac,w)

def erdos_renyi(n,p):
    G = nx.erdos_renyl_graph(n,p)
    nx.draw(G, with_label=True)
    plt.show()

####################################### Law of the iterated logarithm (This theorem is very poorly illustrated in python as it requires infinite memory...)

def LIL(law, parameter, n):
    
    S=[]
    for k in range(3,n+3):
        state, variable, weights = law(parameter, k)
        S.append((sum(variable) - k*expectedValue(law,parameter))/(sqrt(k*variance(law,parameter)))/k)
        
    
    plt.title("Law of the Iterated Logarithm")
    plt.plot([i for i in range(3,n+3)],S, color = 'red')
    plt.plot([i for i in range(3,n+3)], [sqrt(2*log(log(i))/i) for i in range(3,n+3)])
    plt.plot([i for i in range(3,n+3)], [-sqrt(2*log(log(i))/i) for i in range(3,n+3)])

    plt.show()

####################################### Salem-Zygmund theorem: around trigonometric polynomials


def complex_quadrature(func, a, b, **kwargs):
    def real_func(x):
        return scipy.real(func(x))
    def imag_func(x):
        return scipy.imag(func(x))
    real_integral = quad(real_func, a, b, **kwargs)
    imag_integral = quad(imag_func, a, b, **kwargs)
    return (real_integral[0] + 1j*imag_integral[0], real_integral[1:], imag_integral[1:])



def salem_zygmund(n):
    state, a, u = rademacher(n)
    state, b, u = rademacher(n) 
    def S(x):
        val = 0
        for k in range(n):
            val += a[k]*cos(k*x) + b[k]*sin(k*x)
        return scipy.exp(1j*(1/sqrt(n))*val)
    return(complex_quadrature(S,0,2*pi))

##################################################### Urns problems

def beta_normalized(a,b,x):
    return (1/spp.beta(a,b))*(x**(b-1))*(1-x)**(a-1)

def polya_urn(n,m,tries, runs=1000):
    blue_prop = [n/(n+m)]
    red_prop=[m/(n+m)]
    blues=[]

    for k in range(runs):
        red_prop=[m/(n+m)]
        red=m
        blue_prop=[n/(n+m)]
        blue=n
        for i in range(tries):
            k = randint(1, red+blue)
            if k>=blue+1:
                red+=1
            else:
                blue+=1
            red_prop.append(red/(blue+red))
            blue_prop.append(blue/(blue+red))
        blues.append(blue_prop[-1])
    average = (avg(blue_prop) + avg(red_prop))/2

    templist = [i for i in range(tries+1)]
    fig, ax = plt.subplots(1,2)
    ax[0].title.set_text('Polya urns: Proportions for one run')
    ax[1].title.set_text('Limit proportion of blue balls for '+ str(runs) + ' runs')
    ax[0].set_xlabel('kth time picking a ball')
    ax[0].set_ylabel('Proportion')
    ax[0].plot(templist, red_prop, color = 'red')
    ax[0].grid()
    ax[0].plot(templist, [average for i in range(tries+1)], color='green')
    ax[0].plot(templist, blue_prop, color = 'blue')
    ax[1].set_xlabel('Proportion')
    ax[1].set_ylabel('Number of times proportion was reached')
    a = list(np.linspace(0,1,1000))
    
    ax[1].hist(blues,bins=20, facecolor = '#2ab0ff', edgecolor='#169acf', linewidth=0.5, density=True)
    ax[1].plot(a, [beta_normalized(m,n,x) for x in a], color='red')
    ax[1].legend([ 'Beta distribution', 'Distribution of limit proportions of blue balls'])
    ax[0].legend(['Proportion of red balls', 'Mean of means of each proportions', 'Proportion of blue balls'])
    plt.show()

#Notice how the remaining ball in the urn is most likely to be the ball with the number n. This is known as the middle law. (too lazy to explain here, good luck finding it online...)
def middle_law(n,N):
    results=[]
    for i in range(N):
        
        urn = [i for i in range(0,2*n+1)]
        for i in range(n):
    
            a1,b1,c1=sample(urn,3)
            a = min(a1,b1,c1)
            c = max(a1,b1,c1)
            urn.remove(c)
            urn.remove(a)
        results.append(urn[0])
    
    plt.hist(results,bins=30, facecolor = '#2ab0ff', edgecolor='#169acf', linewidth=0.5, density=True)
    plt.xticks(range(0,2*n+1, 5))
    plt.show()





#####################################################
#We show that if (X_n) is a sequence of i.i.d variables, then Y_n = #{X_1,...,X_n} satisfies E(Y_n) = o(sqrt(n)) as n goes to infinity.
#Holds for variables à priori whose values are in the set of positive integers, but may work for others, à fortiori for bounded variables
def cardinal_simulation(law, parameter, n, average_counter=100):
    Y = []
    for i in range(1,n+1):
        Yi=[]
        for k in range(average_counter):
            state, variable, weights = law(parameter, i)
            setv = set(variable)
            Yi.append(len(setv))
        Y.append(avg(Yi)/sqrt(i))
    plt.plot([i for i in range(1,n+1)], Y)
    plt.show()

#Castle duel: there are a certain amount of castles in a line in front of you. You have a certain amount of soldiers. 
#Each castle can give you a specific amount of points (castle i gives you i points). You are aware of the points that each castle gives.
#If you send a1 soldiers to castle j and your enemy sends in a2 soldiers: if a1>a2, you gain j - (a1-a2) points so you need to be careful.
#Your goal is to choose a specific probability distribution for each castle (you may also choose its parameters to optimize your strategy). 
# A certain amount of soldiers will be picked at random for each castle according to that probability distribution.
#After a certain amount of rounds, we see who wins.
#You may only use random variables with positive integer values.
#(WIP)
def castle_duel(law1,parameter1, law2, parameter2, rounds=10,castles=10,soldiers=100):
    results1=[]
    results2=[]
   
    winner1=[]
    winner2=[]
    
    for i in range(rounds):
        u1=[soldiers+1]
        u2=[soldiers+1]
        while sum(u1) > soldiers or sum(u2) > soldiers or (min(u1)<0 or min(u2)<0):
            if sum(u1) > soldiers or min(u1)<0:
                 _,u1,_=law1(parameter1,castles)
            if sum(u2)>soldiers or min(u2)<0:
                _,u2,_=law2(parameter2,castles)
        results1.append(u1)
        results2.append(u2)
        points1=0
        points2=0
        for j in range(castles):
            if u1[j]==u2[j]:
                points1+=0
            elif u1[j]>u2[j]:
                points1+= j+1 - (u1[j]-u2[j])
            else:
                points2+= j+1 - (u2[j]-u1[j])
        if points1>points2:
            winner1.append(1)
        elif points2>points1:
            winner2.append(1)
        else:
            winner1.append(1)
            winner2.append(1)
    
    fig, ax = plt.subplots(rounds, 2)
    for i in range(rounds):
        ax[i][0].bar([i for i in range(1,castles+1)], results1[i])
        ax[i][1].bar([i for i in range(1,castles+1)], results2[i])
    if len(winner1)>len(winner2):
        print("1 Wins")
    elif len(winner2)>len(winner1):
        print("2 Wins")
    else:
        print("Tie")
    plt.show()


###################### Brownian tree (WIP)

import pygame, sys, os
from pygame.locals import *
from random import randint
pygame.init()

MAXSPEED = 15
SIZE = 3
COLOR = (45, 90, 45)
WINDOWSIZE = 400
TIMETICK = 1
MAXPART = 50

freeParticles = pygame.sprite.Group()
tree = pygame.sprite.Group()

window = pygame.display.set_mode((WINDOWSIZE, WINDOWSIZE))
pygame.display.set_caption("Brownian Tree")

screen = pygame.display.get_surface()


class Particle(pygame.sprite.Sprite):
    def __init__(self, vector, location, surface):
        pygame.sprite.Sprite.__init__(self)
        self.vector = vector
        self.surface = surface
        self.accelerate(vector)
        self.add(freeParticles)
        self.rect = pygame.Rect(location[0], location[1], SIZE, SIZE)
        self.surface.fill(COLOR, self.rect)

    def onEdge(self):
        if self.rect.left <= 0:
            self.vector = (abs(self.vector[0]), self.vector[1])
        elif self.rect.top <= 0:
            self.vector = (self.vector[0], abs(self.vector[1]))
        elif self.rect.right >= WINDOWSIZE:
            self.vector = (-abs(self.vector[0]), self.vector[1])
        elif self.rect.bottom >= WINDOWSIZE:
            self.vector = (self.vector[0], -abs(self.vector[1]))

    def update(self):
        if freeParticles in self.groups():
            self.surface.fill((0,0,0), self.rect)
            self.remove(freeParticles)
            if pygame.sprite.spritecollideany(self, freeParticles):
                self.accelerate((randint(-MAXSPEED, MAXSPEED), 
                                 randint(-MAXSPEED, MAXSPEED)))
                self.add(freeParticles)
            elif pygame.sprite.spritecollideany(self, tree):
                self.stop()
            else:
                self.add(freeParticles)
                
            self.onEdge()

            if (self.vector == (0,0)) and tree not in self.groups():
                self.accelerate((randint(-MAXSPEED, MAXSPEED), 
                                 randint(-MAXSPEED, MAXSPEED)))
            self.rect.move_ip(self.vector[0], self.vector[1])
        self.surface.fill(COLOR, self.rect)

    def stop(self):
        self.vector = (0,0)
        self.remove(freeParticles)
        self.add(tree)

    def accelerate(self, vector):
        self.vector = vector

NEW = USEREVENT + 1
TICK = USEREVENT + 2

pygame.time.set_timer(NEW, 50)
pygame.time.set_timer(TICK, TIMETICK)


def input(events):
    for event in events:
        if event.type == QUIT:
            sys.exit(0)
        elif event.type == NEW and (len(freeParticles) < MAXPART):
            Particle((randint(-MAXSPEED,MAXSPEED),
                      randint(-MAXSPEED,MAXSPEED)),
                     (randint(0, WINDOWSIZE), randint(0, WINDOWSIZE)), 
                     screen)
        elif event.type == TICK:
            freeParticles.update()


half = WINDOWSIZE/2
tenth = WINDOWSIZE/10

root = Particle((0,0),
                (randint(half-tenth, half+tenth), 
                 randint(half-tenth, half+tenth)), screen)
root.stop()

while True:
    input(pygame.event.get())
    pygame.display.flip()
