
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