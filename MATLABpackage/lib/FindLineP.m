function [UYcount,UYlines,UYpoints,UYind,UYmask]=...
            FindLineP(PPU,PPY,VF,Td,NumMats)
% Constants
H=768;
Wmasks=[6 5 5 4 4 4 3 3 3];          % weights of masks
Imasks=[1,6; 1,5; 2,6; 2,5; 1,4; 3,6; 3,5; 2,4]; % Start/end positions
nm=length(Wmasks)-1; % (the last weight is artificial) 

UYcount=0; UYlines=0; UYpoints=0; UYind=0; UYmask=0;
C=sum(PPU>0,2); 
PP=2*PPU; % integer points (fiber numbers)
% Prepare map of points
PPmap=false(6,5*H);
PPind=zeros(6,5*H);
for i=1:6
    w=PP(i,1:C(i));
    for j=-2*Td:Td*2
       ww=w+j; ww(ww<=0)=1;
       PPmap(i,ww)=1; PPind(i,ww)=1:C(i);
    end
end
Ucount=0;
for Imask=1:nm
    p=Imasks(Imask,:); 
    %---------------------------------------
    % Draw lines from top to down X
    %---------------------------------------
    % Initialize loop over uppermost and lowermost active points
    a=[0 0];
    if C(p(1))>0
    for i=1:C(p(1))     % start points belong to PP(1,:) or PP(2,:)
        a(1)=PPU(p(1),i);
        if C(p(2))>0
        for j=1:C(p(2)) % loop over final points
            % find intersections with all planes lines 
            a(2)=PPU(p(2),j);
            [mask,Line]=LineWeight(a,PPmap,p,VF);
            count=sum(mask);
            % Update
            if count > max(Ucount,2)
               Ucount=count; 
               Uline=Line;
               Umask=mask;
               if Ucount==Wmasks(Imask), break; end
            end
        end % j
        end % if
        if Ucount==Wmasks(Imask), break; end
    end 
    end
    if Ucount==Wmasks(Imask), break; end
end % LOOP over U configuraions
if Ucount<3, return; end
% recover points and indices using look-up table
w=Uline>=1/2;  Uind=zeros(6,1);
Uind(w)=diag(PPind(w,floor(2*Uline(w))));              
Uind(w)=max([Uind(w) diag(PPind(w,ceil(2*Uline(w))))],[],2);
Upoints=zeros(1,6); 
Umask=Umask&w;
Upoints(Umask)=diag(PPU(Umask,Uind(Umask)));

% ---------------------
% Lai loop (exactly the same!)
%---------------------
C=sum(PPY>0,2); 
PP=2*PPY; % integer points (fiber numbers)
% Prepare map of points
PPmap=false(6,5*H);
PPind=zeros(6,5*H);
for i=1:6
    w=PP(i,1:C(i));
    for j=-Td:Td
       ww=w+j; ww(ww<=0)=1;
       PPmap(i,ww)=1; PPind(i,ww)=1:C(i);
    end
end
Ycount=0;
for Imask=1:nm
    p=Imasks(Imask,:); 
    %---------------------------------------
    % Draw lines from top to down X
    %---------------------------------------
    % Initialize loop over uppermost and lowermost active points
    if C(p(1))>0
    a=[0 0];
    for i=1:C(p(1))     % start points belong to PP(1,:) or PP(2,:)
        if C(p(2))>0
        a(1)=PPY(p(1),i);    
        for j=1:C(p(2)) % loop over final points
            % find intersections with all planes lines 
            a(2)=PPY(p(2),j);
            [mask,Line]=LineWeight(a,PPmap,p,VF);
            count=sum(mask);
            % Update
            if count > max(Ycount,2)
               Ycount=count; 
               Yline=Line;
               Ymask=mask;
               if Ycount==Wmasks(Imask), break; end
            end
        end % j
        end % if
        if Ycount==Wmasks(Imask), break; end
    end 
    end
    if Ycount==Wmasks(Imask), break; end
end % LOOP over U configuraions
if Ycount<3, return; end

% recover points and indices using look-up table
w=Yline>=1/2;  Yind=zeros(6,1);
Yind(w)=diag(PPind(w,floor(2*Yline(w))));              
Yind(w)=max([Yind(w) diag(PPind(w,ceil(2*Yline(w))))],[],2);
Ypoints=zeros(1,6); 
Ymask=Ymask&w;
Ypoints(Ymask)=diag(PPY(Ymask,Yind(Ymask)));

% Output arrays
UYcount=Ucount+Ycount;
UYlines=[Uline;Yline];
UYpoints=[Upoints; Ypoints];
UYind=[Uind Yind];
UYmask=[Umask; Ymask];
% -------------------------------------------------------
function [wa,Line]=LineWeight(a,PPmap,p,VF)
% Given line a=[as af] (start and finish
% compute wa = # of points from PP on this line
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