function [COORD,COORDA,maskID,st,en]=...
    Cloud2EventO(PP,INDhds,Disp,Td,STARTS,ENDS)
% Analysis of Parallel pairs (UD and LR)
% Input:
%   PP   is cloud is 12 x nmax array of point candidates on 12 mats
%   INDhv  is index of pair (1=UD, 6=LR)
%   Disp show picture flag 
%   Td is allowed total deviation between true and extrapolation
%   STARTS,ENDS are time indices for candidates in cloud
% Output: 
%   COORD={X1,Y1,Z1,X2,Y2,Z2} are coordinates of points
%   COORDA={X1,Y1,Z1,X2,Y2,Z2} are linearly approximated 
%                              coordinates of points
%   (xe,ye) are estimated coordinates of muon point 
%   (st,en) are time indices for computing duration of event
%   maskID = -1/0/1  shows information about panels:
%       -1  means that panel is not active
%        0  shows that the panel not used for approximation   
%        1  shows points passed by the ray
%-------------------------------------------------------
if Disp==1, figure(1); clf(1);  end        
COORD=0; COORDA=0;        % Assign values to outputs 
st=zeros(1,6); en=st;     % maskID==2 means 'no result'
% Geometry constants:
% H=768; 
DX=73.5; DZ=103;   % Inner size of HDS camera 
A=PP(1:2:6,:); A(A>0)=A(A>0)+DX; PP(1:2:6,:)=A;     % match with GCS
A=PP(7:2:11,:); A(A>0)=A(A>0)+DZ; PP(7:2:11,:)=A;     % match with GCS
% Search parameters
% Wmasks=[6 5 5 4 4 4 3 3 3];          % weights of masks
% Wmasks=Wmasks(Wmasks>=NumMats);    % reduced list of masks 
% Imasks=[1,6; 1,5; 2,6; 2,5; 1,4; 3,6; 3,5; 2,4]; % Start/end positions
% nm=length(Wmasks)-1; % (the last weight is artificial) 
% if INDhds==4 || INDhds==5
%         I=1:6; J=[5 6 3 4 1 2];
%         PP(I,:)=PP(J,:);            % Flip D(R)-HDS 
%         STARTS(I,:)=STARTS(J,:);  ENDS(I,:)=ENDS(J,:); 
% end

a=PP(:,1); PP(a==0)=1;  % avoid zeros

% Z/X scales depending on muon trajectory
ZF1=[1174 1074 974  -1   -1   -1];   % up
ZF2=[-200 -100  0   -1   -1   -1];   % down
XF1=[-1     -1  -1 915 1015 1115];   % left
XF2=[-1     -1  -1   0 -100 -200];   % right
switch INDhds
    case 2, XF=XF1; ZF=ZF1;   % U->L
    case 3, XF=XF2; ZF=ZF1;   % U->R 
    case 4, XF=XF1; ZF=ZF2;   % D->L
    case 5, XF=XF2; ZF=ZF2;   % D->R
end
PPU=PP(1:2:12,:);       % Puncured pairs for lai
PPY=PP(2:2:12,:);       % Puncured pairs for pikk
[UYcount,UYlines,UYpoints,UYind,UYmask]=FindLineO(PPU,PPY,XF,ZF,Td);

if Disp==1, ShowHits1(INDhds,PPU,UYlines); ShowHits2(INDhds,PPY,UYlines); end

if UYcount<6
    maskID=0; return; 
end

% Form time data
IND=zeros(1,12); 
IND(1:2:11)=UYind(:,1);
IND(2:2:12)=UYind(:,2);
maskID=UYmask;
UYmask=[2 1 0]*UYmask';
st=zeros(1,6); en=zeros(1,6);
for j=1:6  % Time information
    switch UYmask(j)
        case 1  % Only Y
            st(j)= STARTS(2*j,IND(2*j));   % Y 
            en(j)= ENDS(2*j,  IND(2*j));   % Y
        case 2  % Only U
            st(j)= STARTS(2*j-1,IND(2*j-1));   % Y 
            en(j)= ENDS(2*j-1,  IND(2*j-1));   % Y    
        case 3
            st(j)=min([STARTS(2*j-1,IND(2*j-1));   % X
                       STARTS(2*j  ,IND(2*j))]);   % Y 
            en(j)=max([ENDS(2*j-1,  IND(2*j-1));   % X
                   ENDS(2*j  ,  IND(2*j))]);   % Y
    end
end
% Coordinate data 

COORD =UYpoints([1 3 2],:)';
COORDA=UYlines([1 3 2],:)';
% maskID=[Umask' Ymask' Umask'];
maskID(:,1)=maskID(:,1)|(XF'~=-1);
maskID(:,3)=maskID(:,3)|(ZF'~=-1);
return   

%------------------------------------------------------------------------
function ShowHits1(ind,PPU,Line)   % subplot XZ 
% Show active hits XZ projection
C1=sum(PPU>0,2); DX=73.5; DZ=103;
Z0=[1174 1074 974 0 -100 -200];  
X0=[-200 -100 0 915 1015 1115];  
%C1=C(1:2:12,:); % C2=C(2:2:12,:);
subplot(1,2,1);
switch ind
case 2  % UL
    for i=1:6
       if i<=3
          plot([DX,768+DX],[Z0(i) Z0(i)],'--k'); hold on
          if C1(i)>0
           plot(PPU(i,1:C1(i)),Z0(i)*ones(1,C1(i)), '*'); hold on
          end
       else 
          plot([X0(i) X0(i)],[DZ 768+DZ],'--k'); hold on
          if C1(i)>0
           plot(X0(i)*ones(1,C1(i)),PPU(i,1:C1(i)), '*'); hold on
          end
       end
    end
    axis([0,1200,0,1200]);
    xlabel('X'); ylabel('Z');
    title('UL muons, XZ view')
case 3 % UR
    X0=[-1 -1 -1 0 -100 -200];
    for i=1:6
       if i<=3
          plot([DX,768+DX],[Z0(i) Z0(i)],'--k'); hold on
          if C1(i)>0
           plot(PPU(i,1:C1(i)),Z0(i)*ones(1,C1(i)), '*'); hold on
          end
       else 
          plot([X0(i) X0(i)],[DZ 768+DZ],'--k'); hold on
          if C1(i)>0
           plot(X0(i)*ones(1,C1(i)),PPU(i,1:C1(i)), '*'); hold on
          end
       end
    end
    axis([-250,1000,0,1200]);
    xlabel('X'); ylabel('Z');
    title('UR muons, XZ view')
case 4 % LD
    Z0=[-200 -100  0   -1   -1   -1];   % up
    X0=[-1     -1  -1 915 1015 1115];   % left
     for i=1:6
       if i<=3
          plot([DX,768+DX],[Z0(i) Z0(i)],'--k'); hold on
          if C1(i)>0
           plot(PPU(i,1:C1(i)),Z0(i)*ones(1,C1(i)), '*'); hold on
          end
       else 
          plot([X0(i) X0(i)],[DZ 768+DZ],'--k'); hold on
          if C1(i)>0
           plot(X0(i)*ones(1,C1(i)),PPU(i,1:C1(i)), '*'); hold on
          end
       end
     end
    axis([0,1250,-250,1000]);
    xlabel('X'); ylabel('Z');
    title('LD muons, XZ view')
case 5 % RD
    Z0=[-200 -100  0   -1   -1   -1];   % up
    X0=[-1     -1  -1   0 -100 -200];   % right 
    for i=1:6
       if i<=3
          plot([DX,768+DX],[Z0(i) Z0(i)],'--k'); hold on
          if C1(i)>0
           plot(PPU(i,1:C1(i)),Z0(i)*ones(1,C1(i)), '*'); hold on
          end
       else 
          plot([X0(i) X0(i)],[DZ 768+DZ],'--k'); hold on
          if C1(i)>0
           plot(X0(i)*ones(1,C1(i)),PPU(i,1:C1(i)), '*'); hold on
          end
       end
    end
    axis([-250,1000,-250,1000]);
    xlabel('X'); ylabel('Z');
    title('RD muons, XZ view')
end
if nargin==3 && size(Line,2)==6
    plot(Line(1,:),Line(2,:),'-k'); hold on;
end

hold off

%------------------------------------------------------------------------
function ShowHits2(ind,PPY,Line)    % subplot YZ
% Show active hits XZ projection
% C=sum(PP>0,2); 
C2=sum(PPY>0,2);
 subplot(1,2,2);
switch ind
case {2,3}
    Z0=[1174 1074 974 0 -100 -200];  
    for i=1:6
       if i<=3
          plot([0 2*768],[Z0(i) Z0(i)],'--k'); hold on
          if C2(i)>0
           plot(PPY(i,1:C2(i)),Z0(i)*ones(1,C2(i)), '*'); hold on
          end
       else 
          if nargin==3 && size(Line,2)==6
          plot([0 2*768],[Line(2,i) Line(2,i)],'--k'); hold on
          if C2(i)>0
            plot(PPY(i,1:C2(i)),Line(2,i)*ones(1,C2(i)), '*'); hold on
          end
          end
       end
    end
    if nargin==3 && size(Line,2)==6
        plot(Line(3,:),Line(2,:),'-k'); hold on;
    end
    xlabel('Y'); ylabel('Z');
    if ind==2
         title('UL muons, YZ projection'); 
    else
        title('UR muons, YZ projection'); 
    end

    axis([0,1600,0,1200]);
case {4,5}
     Z0=[-200 -100  0   -1   -1   -1];   % up
     for i=1:6
       if i<=3
          plot([0 2*768],[Z0(i) Z0(i)],'--k'); hold on
          if C2(i)>0
           plot(PPY(i,1:C2(i)),Z0(i)*ones(1,C2(i)), '*'); hold on
          end
       else 
           if nargin==3 && size(Line,2)==6
          plot([0 2*768],[Line(2,i) Line(2,i)],'--k'); hold on
          if C2(i)>0
           plot(PPY(i,1:C2(i)),Line(2,i)*ones(1,C2(i)), '*'); hold on
          end
           end
       end
    end
    if nargin==3 && size(Line,2)==6
        plot(Line(3,:),Line(2,:),'-k'); hold on;
    end
    xlabel('Y'); ylabel('Z'); 
    if ind==4
        title('LD muons, YZ projection');
    else
        title('RD muons, YZ projection');
    end

    axis([0,1600,-250,1000]);

 
end
 
hold off