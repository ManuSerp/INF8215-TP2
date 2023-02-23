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


sum(X,Y,R):- R is X+Y.
max2(X,Y,M):- X>=Y, M is X; X<Y, M is Y.
max3(X,Y,Z,M):- max2(X,Y,M1), max2(M1,Z,M).