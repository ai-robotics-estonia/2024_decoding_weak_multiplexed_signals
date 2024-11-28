function [w,wa,Line]=LineWeightP(a,PPmap)
% Given line a=[as af] (start and finish
% global numiter;
% compute wa = # of points from PP on this line
p=[1 6];
VF=[1174 1074 974 0 -100 -200];
a=a*2; 
% draw line a
v=VF(p);
dV=v(1)-v(2);
da=a(1)-a(2); A= da/dV; B=a(1)-A*v(1);
Line=VF*A+B; LineC=ceil(Line); LineF=floor(Line);
LineC(LineC<=0)=1; LineF(LineF<=0)=1;
wa=false(1,6);
for i=1:6
    wa(i)=PPmap(i,LineC(i))|PPmap(i,LineF(i));
end
Line=Line/2;
w=-sum(wa);  %numiter=numiter+1;
% if w~=0
% disp(num2str([numiter, a w wa]))
% wa=abs(wa);
% end
return