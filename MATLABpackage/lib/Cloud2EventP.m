function [COORD,COORDA,maskID,st,en]=...
    Cloud2EventP(PP,INDhv,Disp,Td,STARTS,ENDS)
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
H=768; DX=73.5; DZ=103;   % Inner size of HDS camera 
A=PP(1:2:12,:); A(A>0)=A(A>0)+DX; PP(1:2:12,:)=A;     % match with GCS
PP(PP(:,1)==0,1)=1/2;

% Artificial coordinates U,Y,V instead of X,Y,Z
% One coordinate is known in advance, either Z for UD or X for LR 
T=[0 100 200]; Tf=[200 100 0]; % Coordinates of planes and flipped coord.
switch INDhv
    case 1, VF=[Tf+H+2*DZ -T]; % UD, Z-coordinate 
        I=7:12; J=[5 6 3 4 1 2]+6;
        PP(I,:)=PP(J,:);            % Flip D(R)-HDS 
        STARTS(I,:)=STARTS(J,:);  ENDS(I,:)=ENDS(J,:); 
    case 6 
        VF=[Tf+H+2*DX -T]; % UD, X-coordinate 
end
PPU=PP(1:2:11,:);       % Puncured pairs for pikk
PPY=PP(2:2:12,:);       % Puncured pairs for Y (lai)
[UYcount,UYlines,UYpoints,UYind,UYmask]=...
            FindLineP(PPU,PPY,VF,Td);

if UYcount<6, maskID=0; return; end
if Disp==1
if INDhv==1, ShowHitsH(1,PPU,UYlines); ShowHitsH(2,PPY,UYlines); 
else, ShowHitsV(1,PPU,UYlines); ShowHitsV(2,PPY,UYlines); end
end

% Form time data
IND=reshape(UYind',1,12);
% IND=zeros(1,12); 
% IND(1:2:11)=UYind(1,:);
% IND(2:2:12)=UYind;
maskID=[UYmask' ones(6,1)];
UYmask=[2 1]*UYmask;
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
% Coordinates data 
COORD =[UYpoints' VF']; % [Upoints' Ypoints' VF'];
COORDA=[UYlines'  VF'];  % Yline' VF'];

if INDhv==6 % swap X and Z
    COORD=COORD(:,[3 2 1]); 
    COORDA=COORDA(:,[3 2 1]); 
    maskID=maskID(:,[3 2 1]);
end

return   

%------------------------------------------------------------------------
function ShowHitsH(mode,PP,Line)
% Show active hits 
% figure(1);clf(1);
subplot(1,2,mode); 
C=sum(PP>0,2);
Z=[1174 1074 974 0 -100 -200];  
for i=1:6
    % Draw panels for Z=-200,-100,0,..., H+2*DZ
    plot([0,2000],[Z(i) Z(i)],'--k'); hold on
    % Draw x-points on the current line
    if C(i)>0
        plot(PP(i,1:C(i)),Z(i)*ones(1,C(i)), '*'); hold on
    end
end
if mode == 1 
    axis([0 1000 -250 1250 ]); title('UD muons, XZ view');
    xlabel('X'); ylabel('Z');
else 
    axis([0 1500 -250 1250 ]); title('UD muons, YZ view');  
    xlabel('Y'); ylabel('Z');
end
if nargin==2, return; end
subplot(1,2,mode), plot(Line(mode,:),Z,'-ok');  hold on
return
%------------------------------------------------------------------------
function ShowHitsV(mode,PP,Line)
% Show active hits 
% figure(1);clf(1);
subplot(1,2,mode); 
C=sum(PP>0,2);
% Z=[1174 1074 974 0 -100 -200];  
Z=[1115 1015 915 0 -100 -200];  
for i=1:6
    % Draw panels for Z=-200,-100,0,..., H+2*DZ
    plot([Z(i) Z(i)],[0,2000],'--k'); hold on
    % Draw x-points on the current line
    if C(i)>0
        plot(Z(i)*ones(1,C(i)),PP(i,1:C(i)), '*'); hold on
    end
end
if mode == 1 
    axis([ -250 1250 0 1000]); title('LR muons XZ view');
    xlabel('X'); ylabel('Z');
else 
    axis([ -250 1250 0 1500]); title('LR muons YZ view');  
    xlabel('X'); ylabel('Y');
end
if nargin==1, return; end
subplot(1,2,mode), plot(Z,Line(mode,:),'-ok');  hold on
return