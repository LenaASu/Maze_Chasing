% Basics
palindrome(L):-reverse(L).

list_range(X,Y,[]):-
    X = Y.
list_range(X,Y,[X|N]):-
    X < Y,
    M is X+1,
    list_range(M,Y,N).
list_range(X,Y,[X|N]):-
    X > Y,
    M is X-1,
    list_range(M,Y,N).


prime(2).
prime(X):-
    X > 2,
    \+ mod_zero(X,2).

mod_zero(X,M):-
    X mod M =:= 0.
mod_zero(X,M):-
    M * M < X,
    NextM is M+1,
    mod_zero(X,NextM).

prime_range(X,Y,[]):-
    X = Y.

prime_range(X,Y,[X|N]):-
    X < Y,             
    prime(X),
    M is X+1,      
    prime_range(M,Y,N).

prime_range(X,Y,N):-
    X < Y,              
    \+ prime(X),
    M is X+1,      
    prime_range(M,Y,N).

prime_range(X,Y,[X|N]):-
    X > Y,
    M is X-1,
    prime(M),
    list_range(M,Y,N).

prime_range(X,Y,N):-
    X > Y,              
    \+ prime(X),
    M is X-1,      
    prime_range(M,Y,N).
 

% Simple Maze Solving
connected(1,2).
connected(3,4).
connected(5,6).
connected(7,8).
connected(9,10).
connected(11,13).
connected(13,14).
connected(15,16).
connected(17,18).
connected(19,20).
connected(4,1).
connected(6,3).
connected(4,7).
connected(6,12).
connected(14,9).
connected(12,15).
connected(16,11).
connected(14,17).
connected(16,19).

path(X,Y):-
    connected(X,Y).
path(X,Y):-
    connected(X,M),
    path(M,Y).
    
path_length(X,Y,0):-
    X = Y.

path_length(X,Y,1):-
    connected(X,Y).

path_length(X,Y,N):-
    connected(X,M),
    path_length(M,Y,Prev),
    N is Prev+1.


% Graph Cycles
edge(N1,N2):-
    connected(N1,N2).
edge(N1,N2):-
    connected(N1,M),
    edge(M,N2).

edge_path(N1, N2, [connected(N1,N2)]):-
    connected(N1, N2).

edge_path(N1,N2,P):-
    connected(N1,M),
    edge_path(M,N2,Prev),
    P = [connected(N1,M)|Prev].
    
cycle(A,P):-
    connected(A,M),
    edge_path(M,A,Prev),
    P = [connected(A,M)|Prev].
    