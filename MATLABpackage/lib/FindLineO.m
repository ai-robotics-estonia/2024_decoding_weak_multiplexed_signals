function [UYcount,UYlines,UYpoints,UYind,UYmask]=...
            FindLineO(PPU,PPY,XF,ZF,Td)
H=768;
Wmasks=[6 5 5 4 4 4 3 3 3];          % weights of masks
Imasks=[1,6; 1,5; 2,6; 2,5; 1,4; 3,6; 3,5; 2,4]; % Start/end positions
nm=length(Wmasks)-1; % (the last weight is artificial) 


UYcount=0;UYlines=0; UYpoints=0; UYind=0; UYmask=0;
CU=sum(PPU>0,2); CY=sum(PPY>0,2); 
PP=2*PPU; % integer points (fiber numbers)
% Prepare map of points:  
PPUmap=false(6,6*H);
PPUind=zeros(6,6*H);
for i=1:6
    w=PP(i,1:CU(i));
    for j=-2*Td:Td*2
       ww=w+j; ww(ww<=0)=1;
       PPUmap(i,ww)=1; PPUind(i,ww)=1:CU(i);
    end
end
PP=2*PPY; % integer points (fiber numbers)
% Prepare map of points:  
PPYmap=false(6,6*H);
PPYind=zeros(6,6*H);
for i=1:6
    w=PP(i,1:CY(i));
    for j=-2*Td:Td*2
       ww=w+j; ww(ww<=0)=1;
       PPYmap(i,ww)=1; PPYind(i,ww)=1:CY(i);
    end
end

Ucount=0; Ycount=0;
for Imask=1:nm
    p=Imasks(Imask,:);  % start/end mats
    %---------------------------------------
    % Draw lines between X and Z
    %---------------------------------------
    % Initialize loop over uppermost and lowermost active points
    a=[0 0];
    ucount=0;
    if CU(p(1))>0
    for i=1:CU(p(1))      % start points belong to PP(1,:) 
        a(1)=PPU(p(1),i);    % start point
        if CU(p(2))>0                 
        for j=1:CU(p(2)) % loop over line end points
            % disp([i,j])
            % find intersections with all planes lines 
            a(2)=PPU(p(2),j);    % start point
            [mask,Line]=LineWeightU(a,PPUmap,p,2*XF,2*ZF);
            % 2-points extrapolation
            count=min(sum(mask,2));
            % Update
            if count > max(ucount,2)
               ucount=count; 
               uline=Line;
               umask=mask;
               % if Disp==1  
               %     ShowHits1(INDhds,PP,Line); 
               % end
               % Y-loop: Z-line is known, search for best y-points
               % count=0;
               for IImask=1:nm
                  py=Imasks(IImask,:); 
                  if CY(py(1))>0
                  ay=[0 0];
                  for iy=1:CY(py(1))     % loop over start points 
                     if CY(py(2))>0
                     ay(1)=PPY(py(1),iy);
                     for jy=1:CY(py(2)) % loop over final points 
                        ay(2)=PPY(py(2),jy);
                        [mask,yline]=LineWeightY(ay,PPYmap,py,Line(2,:));
                        count=sum(mask);
                        if count >  max(Ycount, 2) ...
                           && ucount+count>UYcount
                           Ycount=count; 
                           Ucount=ucount;
                           Uline=uline;
                           Yline=yline;
                           Umask=umask;
                           UYcount=Ucount+Ycount; 
                           Ymask=mask; 
                           if Ycount==Wmasks(IImask), break; end
                        end
                     end % jy
                     end % CY(p(2))>0
                     if UYcount>=6, break; end
                  end % CY(p(1))>0 
                  end % iy
                  if Ycount==Wmasks(IImask+1), break; end
               end % y-mask
            end  % if XZ updated
            if UYcount>=6, break; end
        end % j
        if UYcount>=6, break; end
        end % if
        % if Ycount==Wmasks(Imask), break; end
    end % i
    if UYcount>=6, break; end
    end % if C
    if UYcount>=6, break; end
end %masks

if Ycount<3, return; end
% recover U-points and indices using look-up table
Uind=zeros(6,1); w=[Uline(1,1:3) Uline(2,4:6)]>=1/2;
Upoints=Uline;
Umask=Umask&w;
for i=1:3
    if Umask(i)
        Uind(i)=max(PPUind(i,floor(2*Uline(1,i))),...
                    PPUind(i,ceil(2*Uline(1,i))));
        Upoints(1,i)=PPU(i,Uind(i));
    else
        Upoints(1,i)=0;
    end
    if Umask(i+3)
        Uind(i+3)=max(PPUind(i+3,floor(2*Uline(2,i+3))), ...
                      PPUind(i+3,ceil(2*Uline(2,i+3)))); 
        Upoints(2,i+3)=PPU(i+3,Uind(i+3));
    else 
        Upoints(2,i+3)=0;
    end
end


% recover Y-points and indices using look-up table
w=Yline>=1/2;  Yind=zeros(6,1); 
Yline(Yline>2*H)=2*H;
Yind(w)=diag(PPYind(w,floor(2*Yline(w))));              
Yind(w)=max([Yind(w) diag(PPYind(w,ceil(2*Yline(w))))],[],2);
Ypoints=zeros(1,6); 
Ymask=Ymask&w;
Ypoints(Ymask)=diag(PPY(Ymask,Yind(Ymask)));

% Output arrays
% UYcount=Ucount+Ycount;
UYlines=[Uline;Yline];
UYpoints=[Upoints; Ypoints];
UYind=[Uind Yind];
UYmask=[Umask; Ymask; Umask]';
return;



% -------------------------------------------------------
function [wa,Line]=LineWeightU(a,PPmap,p,XF,ZF)
% Given line a=[as af] (start and finish
% compute wa = # of points from PP on this line
a=a*2; 
% draw line a
dz=ZF(p(1))-a(2);
dx=a(1)-XF(p(2));
Axz= dz/dx; Bxz=ZF(p(1))-Axz*a(1);
Azx= 1/Axz; Bzx=-Bxz/Axz;

u=Azx*ZF(1:3)+Bzx; u(u<=1)=1; u(u>2348)=3000; uc=ceil(u); uf=floor(u);
v=Axz*XF(4:6)+Bxz; v(v<=1)=1; v(v>2348)=3000; vc=ceil(v); vf=floor(v);

% LineC=[uc XF(4:6);  ZF(1:3) vc]; 
% LineF=[uf XF(4:6);  ZF(1:3) vf]; 
% wa=false(1,6);
% for i=1:3
%     wa(i)=PPmap(i,LineC(1,i))|PPmap(i,LineF(1,i));
%     wa(i+3)=PPmap(i+3,LineC(2,i+3))|PPmap(i+3,LineF(2,i+3));
% end

wa=false(1,6);
for i=1:3
    wa(i)=PPmap(i,uc(i))|PPmap(i,uf(i));
    wa(i+3)=PPmap(i+3,vc(i))|PPmap(i+3,vf(i));
end


Line=[u XF(4:6);  ZF(1:3) v]/2;

% -------------------------------------------------------
function [wa,Line]=LineWeightY(a,PPmap,p,VF)
% Given line a=[as af] (start and finish
% compute wa = # of points from PP on this line
a=a*2; 
% draw line a
v=VF(p);
dV=v(1)-v(2);
da=a(1)-a(2); A= da/dV; B=a(1)-A*v(1);
Line=VF*A+B; LineC=ceil(Line); LineF=floor(Line);
LineC(LineC<=0)=1; LineF(LineF<=0)=1;
LineC(LineC>3072)=3072; LineF(LineF>3072)=3072;
wa=false(1,6);
for i=1:6
    wa(i)=PPmap(i,LineC(i))|PPmap(i,LineF(i));
end
Line=Line/2;