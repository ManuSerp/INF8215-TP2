
% 2229693_1958185_1828113
%Exercice 3.7

homme(hugo).
homme(gabriel).
homme(alexis).
homme(loic).
homme(maxime).
homme(mathieu).
femme(catherine).
femme(lea).
femme(alice).
femme(rose).
femme(emma).
femme(justine).
parent(hugo,lea).
parent(hugo,gabriel).
parent(catherine,lea).
parent(catherine,gabriel).
parent(gabriel,alexis).
parent(gabriel,rose).
parent(gabriel,emma).
parent(alice,alexis).
parent(alice,rose).
parent(alice,emma).
parent(loic,alice).
parent(loic,maxime).
parent(loic,mathieu).
parent(justine,alice).
parent(justine,maxime).
parent(justine,mathieu).
%rules
enfant(X,Y):- parent(Y,X).
fille(X,Y):- parent(Y,X),femme(X).
fils(X,Y):- parent(Y,X),homme(X).
mere(X,Y):- parent(X,Y),femme(X).
pere(X,Y):- parent(X,Y),homme(X).
frere(X,Y):- parent(Z,X),parent(Z,Y),homme(X),X\=Y.
soeur(X,Y):- parent(Z,X),parent(Z,Y),femme(X),X\=Y.
oncle(X,Y):- parent(Z,Y),frere(X,Z).
tante(X,Y):- parent(Z,Y),soeur(X,Z).
grandparent(X,Y):- parent(X,Z),parent(Z,Y).
grandmere(X,Y):- grandparent(X,Y),femme(X).
grandpere(X,Y):- grandparent(X,Y),homme(X).
petitenfant(X,Y):- grandparent(Y,X).
petitefille(X,Y):- petitenfant(X,Y),femme(X).
petitfils(X,Y):- petitenfant(X,Y),homme(X).

%Exercice 4.3

sum(X,Y,R):- R is X+Y.
max2(X,Y,M):- X>=Y, M is X; X<Y, M is Y.
max3(X,Y,Z,M):- max2(X,Y,M1), max2(M1,Z,M).


%Exercice 4.5

lettre(abalone,a,b,a,l,o,n,e).
lettre(abandon,a,b,a,n,d,o,n).
lettre(enhance,e,n,h,a,n,c,e).
lettre(anagram,a,n,a,g,r,a,m).
lettre(connect,c,o,n,n,e,c,t).
lettre(elegant,e,l,e,g,a,n,t).


h1v1(H,V) :- lettre(H,_,A,_,_,_,_,_),lettre(V,_,B,_,_,_,_,_),A=B,H\=V.

h1v2(H,V) :- lettre(H,_,_,_,A,_,_,_),lettre(V,_,B,_,_,_,_,_),A=B,H\=V.

h1v3(H,V) :- lettre(H,_,_,_,_,_,A,_),lettre(V,_,B,_,_,_,_,_),A=B,H\=V.


h2v1(H,V) :- lettre(H,_,A,_,_,_,_,_),lettre(V,_,_,_,B,_,_,_),A=B,H\=V.

h2v2(H,V) :- lettre(H,_,_,_,A,_,_,_),lettre(V,_,_,_,B,_,_,_),A=B,H\=V.

h2v3(H,V) :- lettre(H,_,_,_,_,_,A,_),lettre(V,_,_,_,B,_,_,_),A=B,H\=V.

h3v1(H,V) :- lettre(H,_,A,_,_,_,_,_),lettre(V,_,_,_,_,_,B,_),A=B,H\=V.
h3v2(H,V) :- lettre(H,_,_,_,A,_,_,_),lettre(V,_,_,_,_,_,B,_),A=B,H\=V.
h3v3(H,V) :- lettre(H,_,_,_,_,_,A,_),lettre(V,_,_,_,_,_,B,_),A=B,H\=V.

crossword(V1,V2,V3,H1,H2,H3) :- h1v1(H1,V1),h1v2(H1,V2),h1v3(H1,V3),h2v1(H2,V1),h2v2(H2,V2),h2v3(H2,V3),h3v1(H3,V1),h3v2(H3,V2),h3v3(H3,V3).






%Exercice 4.8

longueur([],0).
longueur([_|T],N):- longueur(T,N1), N is N1+1.

max([A],A).
max([A|T],M):- max(T,M1),max2(M1,A,M).

somme([],0).
somme([A|T],S):- somme(T,S1), S is S1+A.



nth(N,[A|T],R,S):- S==N, R is A ; S1 is S+1, nth(N,T,R,S1).
nth(N,L,R):- nth(N,L,R,1).

zip([A1],[A2],T):- T=[[A1,A2]].
zip(L1,L2,R):- [A|T1]=L1, [B|T2]=L2, R=[[A,B]|T], zip(T1,T2,T).



enumerate(1,L,C):- L=[C].
enumerate(N,L,C):- N1 is N-1,L=[C|A],C1 is C+1,enumerate(N1,A,C1).
enumerate(N,L):- enumerate(N,L,0).


