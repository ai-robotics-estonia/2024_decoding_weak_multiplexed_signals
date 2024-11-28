function EVENTS=stream2events(fname,Ts,Tq,EQmax,...
         Tr_mode,Td,DISP,Trajectory)
% fname    is data stream file name
% Ts       is search window (event length limit)
% Tq       is sample matching precision for quads
% Tr_mode  is transform mode
% EQmax    is quantization error for candidates in clouds
% Td(1:2)  are admissible extrapolation errors
% DISP     is flag=0/1 for drawing plots
%--------------------------------------------
% Constants:
if nargin<9, Trajectory=ones(1,6); end
clf;            % clear all figures
% fig1          % rays
% fig2          % UD hits and estimates
% fig3          % Histogram of # of measurements/episode
debug=0;        % Intermediate data
DX=73.5; % DZ=106; % Geometrical bias 
MaxCS = 200;    % Maximum Cloud size
udlr='UDLR';
% Detector IDs for pairs of HDS
IDS=[ 1  2  3  4  5  6;   % UD
      1  2  3  9  8  7;   % UL
      1  2  3 12 11 10;   % UR
      6  5  4  9  8  7;   % DL
      6  5  4 12 11 10;   % DR
      7  8  9 12 11 10]'; % LR
% Starts/ends windows for pairs of HDS
WSE=[1:12; 1:6 13:18; 1:6 19:24; 7:18;7:12 19:24;13:24]; 
% Create table: Measurements --> hds, Muxes, etc.
[hds,Muxes,Plates,Mats,Coords,MPind] = LocInfo;
MinRun= 12;% at least 2 measurements/quad both for pikk and lai 
scale=[0 13 23 32 39 49 57 67 80]+1; % quantization scale (nonumiform)
QTABLE=qtable(scale);                % quantization table
% QTABLE1=qtable1(scale);                % quantization table
if DISP(7)>0,  figure(2); clf(2);    end    % Prepare figure 
%--------------------------------------------
% Load and brush TOM data
load(fname,"Tom");            % load data
Tom=sortrows(Tom,1);          % sort Time stamps  
MATS=Mats(Tom(:,3)+1);        % mat numbers
Tom=Tom(MATS>0,:);            % Exclude non-existing fibers 
N=size(Tom,1);                % Number of measurements 
XY=Tom(:,2);                  % read measurements
Tom(XY<0,2)=0; Tom(XY>8,2)=8; % dynamic range restricting 
t=(Tom(N,1)-Tom(1,1))*1e-12;         % Time in s
disp(['Record length in seconds =', num2str(t),' s']);

% measuring inter-sample time intervals
DT=Tom(2:N,1)-Tom(1:(N-1),1); % time durations between measurements
FS=find(DT>Ts);             % Expected  starts of episodes
NS=length(FS);              % Number of starts of episodes
RLS=FS(2:NS)-FS(1:NS-1);    % Lengths of episodes in samples 

% *******  Episodes should be long enough ********
FFS=find(RLS>=MinRun);      % min # of samples in an episode 
FS=FS(FFS);                 % updated list of starts
RLS=RLS(FFS);               % updated run lengths
NS=length(FS);              % updated # of starts

% ******* Delete data outside potential episodes *******
mask=zeros(1,N,'int8');
for i=1:NS
    mask(FS(i)+1)=2;          % Start of run
    mask(FS(i)+2:FS(i)+RLS(i))=1; % intermediate measurements 
end
Tom=Tom(mask>0,:); mask=mask(mask>0); % delete samples
NN=size(Tom,1);             % new number of samples
FS=find(mask==2);           % Expected starts of episodes 
clear FFS;          % Free memory

%--------------------
% Read info 
%--------------------
TC=Tom(:,1); XY=Tom(:,2); Loc=Tom(:,3); % Time, Sample, Location   
HDS=hds(Loc+1);                     % HDS of measurements
COORD=int8(Coords(Loc+1));          % classification x+,x-,y+,y-
MPI=MPind(Loc+1);                   % MPind=mod(2*Muxes,5)+Plates*256;
Index=Plates(Loc+1)*4+Muxes(Loc+1); % joint indexing mux+muxplate
MATS=Mats(Loc+1);              % mat numbers
clear Loc;

fprintf('---------------------\n') 
fprintf('Total number of measurements      = %d\n', N);
fprintf('Number of measurements to process = %d\n', NN);
fprintf('Number of intervals with at least %d measurements = %d\n',...
    MinRun,NS);
fprintf('---------------------\n') 
% Arrays for statistical analysis
NumQuadsPerHDS=zeros(1,1500);        % # of catch muons  
NumHitsPerHDScloud=zeros(1,1500);    % propagation due to list quantization 
NumHDS=0;                           % Total number of HDS processed
NumHitsByMuons=zeros(1,20);        % Number of meaured XYZ 
XYstat=zeros(1,N); Jxy=0;


% -----------------------------------------------
% Filter 2: At least MinHits mats in 2 hodoscopes 
%           and SumHits mats in total
%------------------------------------------------
mask=zeros(NS,4); % mask for hds with both x and y activity
for i=1:NS-1 % over starts 
    w=FS(i)+1:FS(i)+RLS(i); % episode window
    stat=zeros(4,6);        % statistic for hds (rows) and mats (columns)    
    for I=1:4               % over hodoscopes
        for J=1:6           % over mats in hodoscope
            stat(I,J)=sum(HDS(w)==I & MATS(w)==J);
        end
    end
    stat=stat>=2;             % at least 2 coordinate per quad
    statx=sum(stat(:,[1 3 5]),2)>=1;   % pikk (x) 
    staty=sum(stat(:,[2 4 6]),2)>=1;   % lai  (y)
    mask(i,:)=(statx&staty)'; % both  X and Y are active 
end 
summask=sum(mask,2);          % Number of HDS in each episode
% Filter out non-complete episodes 
FFS=find(summask>1);
FS=FS(FFS);                   % updated list of starts
RLS=RLS(FFS);                 % updated run lengths
NS=length(FS);                % updated # of starts
mask=mask(FFS,:);             % active HDS in each full episode

if any(DISP) % Detailed statistics of episodes
    binmask=mask*[8 4 2 1]';  % Characterization of each episode
    histbinmask=zeros(1,15);  % Statistical data for HDS in episodes
    for i=1:15, histbinmask(i)=sum(binmask==i); end
    fprintf('Activity of HDS and mats given NumHits and SumHits\n')
    fprintf('Episodes(hds):')
    for i=1:4  % print statistics
        fprintf('%d(%c);', sum(summask==i),udlr(i));
    end
    fprintf('\n');    
    disp(['Number of episodes with at least two HDS=', num2str(NS),';']);
    if DISP(7)>1, figure(3); histogram(RLS,1:10:700); grid on; 
        title(['Distribution of number '; 'of samples in episodes  ']); 
    end
    fprintf('Number of UD candidates = %d,',histbinmask(12) ); 
    fprintf('UL = %d,',  histbinmask(10) ); 
    fprintf('UR = %d,',  histbinmask( 9) ); 
    fprintf('LD = %d,\n',histbinmask( 6) );
    fprintf('RD = %d,',  histbinmask( 5) );
    fprintf('LR = %d,',  histbinmask( 3) );
    fprintf('UDL+UDR = %d,',histbinmask(14)+histbinmask(13) );
    fprintf('ULR+DLR = %d,',histbinmask(11)+histbinmask( 7) );
    fprintf('UDLR = %d\n',  histbinmask(15));
    fprintf('---------------------\n') 
end
%-----------------------
% Loop over episodes 
%-----------------------
% Memory allocation:
EVENTS=zeros(NS*6,12);       % Format: # , X, Y, Z, time, MAT Index 
we=0;                        % EVENTS index initializations
NE=0;                        % Number of detected episodes
% NQ=0;                      % Number of quadruples
Iudlr=zeros(1,6);            % Counters
% Tables for registration of samples
% (6 (pikk)+ 12(lai) muxplates) * 3 planes = 54 possible muxplates
% Each of 54 muxplate containes 4 muxes, 4 coordinates for a sample 
TestTable=zeros(54*4,4);     % registration of samples
SumTest=zeros(1,54*4);       % accumulation of samples for a quad
nump=0; numc=0;
% express-analysis: If sumtest==4, we have a quadruple 
for i=1:NS %100%  % Over all episodes
    % fprintf('episode=%d \n',i);
    maskw=mask(i,:);         % current list of active hodoscopes
    w=FS(i)+1:FS(i)+RLS(i);  % episode window
    hdsw=HDS(w);             % hodoscope for each sample of episode
    qNt=floor(RLS(i)/4);     % expected number of quads 
    % temporary arrays for one episode
    qINDt=-ones(qNt,2);      % time indices
    qHDSt=-ones(qNt,1);      % hds numbers   
    qMATSt=-ones(qNt,1);     % mat numbers
    qXYt=-ones(qNt,2);       % Transformed x,y
    qENt=-ones(qNt,1);       % Energy
    qMPIt=-ones(qNt,1);      % muxplate&mux index
    nt=0;                    % counter of quads
    HDS_MAT=zeros(4,6);      % distribution of quads over HDSs and MATSs 
    HDS_TIME=-ones(4,6);     % start times of mats
    fmask=find(maskw>0);     % mask of active hds
    %-------------------------
    % Loop over hds in episode
    %-------------------------
    for ihds=fmask             % hodoscope index
        fhds=find(hdsw==ihds); % positions in window
        wh=w(fhds);            % subwindow of given hodoscope
        indh=Index(w(fhds));   % muxplate&mux for each sample 
        %-----------------------------
        % Detect quadruples
        %-----------------------------
        % Find measurements belonging to the same quadruple
        for ii=1:length(wh)    % over hds-subwindow         
            iii=indh(ii);      % index of the sample in the table 
            wii=wh(ii);        % true index 
            TestTable(iii,COORD(wii))=wii; % time position
            SumTest(iii)=SumTest(iii)+1;   % # of detected coord
        end
        % wthr is min allowed # for registering a quad
        % First, register 4-sample quads, then 3 and 2
        for wthr=4:-1:2      % # of active measurements in a quad group 
            f=find(SumTest==wthr);  % active muxplate&mux
            for ii=1:length(f)      % over active muxplate&mux
                iii=f(ii);          % group number
                tt=TestTable(iii,:);% time positions of samples in window
                xy=[0 0 0 0];  % zeros on non-sampled positions         
                xy(tt>0)=XY(tt(tt>0)); % read samples
                [qxy,F,EN]=transform(xy,Tr_mode,Tq);
                if F>0
                    XYstat(Jxy+1:Jxy+length(qxy))=qxy;
                    Jxy=Jxy+2;
                end

                if F>0
                    nt=nt+1;        % number of quads in current mat
                    mint=min(tt(tt>0));      % quad start index
                    maxt=max(tt);            % quad last index
                    qINDt(nt,:)=[mint maxt]; % time indices
                    qHDSt(nt,:)=ihds;        % hds numbers   
                    qMATSt(nt)=MATS(mint);   % mat numbers
                    qXYt(nt,:)=qxy;          % Transformed x,y
                    qENt(nt)=EN;             % Energy
                    qMPIt(nt)=MPI(mint);     % muxplate&mux index
                    HDS_MAT(ihds,MATS(mint))=HDS_MAT(ihds,MATS(mint))+1;
                    if HDS_TIME(ihds,MATS(mint))<0  % not appeared
                        HDS_TIME(ihds,MATS(mint))=mint;% write time
                    end
                end
            end
        end
        % Refresh tables for next quadruple
        TestTable(:,:)=0;
        SumTest(:)=0;
        maskw(ihds)=nt; % qumulative # of quadruples
        if nt>0
            NumQuadsPerHDS(nt)=NumQuadsPerHDS(nt)+1;
        end
   end  % hodoscop processing
   %------------------------------------------------------------
   % Quadruples are formed. Test for NumHits and SumHits
   %------------------------------------------------------------
   % Both Pikk and Lai must be presented in each HDS and MAT 
   % Mask of valid mats
   s=sum(HDS_MAT>0,2);                 % # of hits in each HDS 
   num_hds=sum(s>0);                   % # of active HDS
   hds_mat=HDS_MAT(s>0,:);
   NumMatsX=sum(hds_mat(:,[1 3 5])>0,2);
   NumMatsY=sum(hds_mat(:,[2 4 6])>0,2);
   minMats=min(min(NumMatsY+NumMatsX));  
   sumX=sum(NumMatsX);
   sumY=sum(NumMatsY);
   
    NumHDS=NumHDS+length(fmask);             % 

   
   if minMats>=1 ...         % min number of hits
       && min(sumX,sumY) >= 3 ...   % at least 3 points in X and Y
       && num_hds>=2  % Good episode if at least 2 hodoscopes are active
        % ---------------------
        % Cloud generating
        % ---------------------
        % For 4 HDS and 6 mats find at most MaxCS candidates for CLOUD  
        CLOUDS=-ones(24,MaxCS);     % Memory for clouds
        STARTS=-ones(24,MaxCS);     % Time starts of candidates
        ENDS=-ones(24,MaxCS);       % Time ends of candidates
        for ihds=fmask              % hodoscope index
            %-------------
            % Get cloud
            %-------------
            fhds=find(qHDSt==ihds); % quadruples from current HDS
            inds=qINDt(fhds,:);     % indices of quads
            mats=qMATSt(fhds);      % mats of quads
            mpi=qMPIt(fhds);        % mux-plate indices
            eng=qENt(fhds);         % energies
            xy=qXYt(fhds,:);        % values 
            % Prepair arrays for 6 mats of current HDS
            X=-ones(6,nt); Y=-ones(6,nt); M=-ones(6,nt); 
            IS=-ones(6,nt); IE=-ones(6,nt); E=-ones(6,nt); 
            for imat=1:6
                fmat=mats==imat;    % positions of imat
                nmat=sum(fmat);     % quad counter in the mat 
                if nmat>0
                    X(imat,1:nmat)=xy(fmat,1); 
                    Y(imat,1:nmat)=xy(fmat,2);
                    M(imat,1:nmat)=mpi(fmat); % mux-plate indices
                    E(imat,1:nmat)=eng(fmat);  % energy
                    IS(imat,1:nmat)=inds(fmat,1); % starts
                    IE(imat,1:nmat)=inds(fmat,2); % ends
                end
            end
            [ListCoord,~,JS,JE]=... % List of fibers
                  GetCloud(M,X,Y,IS,IE,ihds,EQmax,QTABLE,E);
            nm=size(ListCoord,2);      % cloud size
            wm=(ihds-1)*6+(1:6);       % window location
            CLOUDS(wm,1:nm)=ListCoord; % write to cloud
            STARTS(wm,1:nm)=JS;        % time
            ENDS(wm,1:nm)=JE;          % info  
            hits=sum(sum(ListCoord>0));
            if hits>0
               NumHitsPerHDScloud(hits)=NumHitsPerHDScloud(hits)+1;
            end
        end
        
        %------------------------------------------
        % Cloud processing (finding straight lines)
        %------------------------------------------
        for i1=1:3               % first active hds of the pair
          if maskw(i1)>0 || debug
          w1=(i1-1)*6+(1:6);   % window in cloud
            for j1=i1+1:4        % second active hds of the pair         
                if maskw(j1)>0 ||debug
                    %------------------------------
                    % Search for line through 2 HDS
                    %------------------------------
                    w2=(j1-1)*6+(1:6);  % window in cloud
                    PAIR=CLOUDS([w1,w2],:); % Non-empty pair of clouds 
                    % Re-Check SumHits and NumHits conditions 
                    C=sum(PAIR>0,2)>0; 
                    s1=sum(C(1:2:12,:)); s2=sum(C(2:2:12,:)); % # of mats
                    minhits=min(s1,s2);      % min # of active planes
                    if minhits>=3
                    pairID=2*i1+j1-3-(2*i1+j1==10);  %(i1,j1)-->{1...6}          
                    maskP=0;  wse=WSE(pairID,:);
                    if Trajectory(pairID)==1
                    switch pairID% (
                        case {1,6}   % <--------- UD    
                           % disp(i)
                           [XYZ,XYZA,maskP,st,en]=...
                              Cloud2EventP(PAIR,pairID,DISP(pairID),...
                           Td(1),STARTS(wse,:),ENDS(wse,:));
                           % ID=1:6;
                        case {2,3,4,5}  % <-------- UL  
                           % disp(i)
                          [XYZ,XYZA,maskP,st,en]=...
                              Cloud2EventO(PAIR,pairID,DISP(pairID),...
                          Td(2),STARTS(wse,:),ENDS(wse,:));
                    end % switch ray direction
                    end
                    
                    if sum(sum(maskP))>=3 % success 
                         % Statistics:
                         numhits=sum(sum(maskP(1:3,1:2)));% 1st hds 
                         if pairID==6, numhits=sum(sum(maskP(1:3,2:3))); end
                         NumHitsByMuons(numhits)=NumHitsByMuons(numhits)+1;
                         numhits=sum(sum(maskP(4:6,1:2)));% 2nd hds 
                         if pairID==6, numhits=sum(sum(maskP(4:6,2:3)));end
                         NumHitsByMuons(numhits)=NumHitsByMuons(numhits)+1;
                         %---------------------------
                         ID=IDS(:,pairID);
                         % disp(i)
                         NE=NE+1; Iudlr(pairID)=Iudlr(pairID)+1;
                         w=we+(1:6); we=we+6; % counters
                         EVENTS(w,1)=NE;
                         EVENTS(w,2:4)=XYZ;
                         EVENTS(w,5:7)=XYZA;
                         EVENTS(w,8:10)=maskP;
                         EVENTS(w(en+st>0),11)=... 
                             double((TC(en(en>0))-TC(st(en>0))))/1000;
                         %fprintf('cloud=%d, UD=%d, among=%d\n',i,Iud,NE);
                         EVENTS(w,12)=ID;
                         f4=find(ID==4,1);
                         if DISP(7)==1 && ~isempty(f4) %ID(4) && pairID==1
                             figure(2); 
                             xe=XYZA(f4,1)-DX; ye=XYZA(f4,2);
                             if maskP(f4)>0 && XYZ(f4,1)>0 && XYZ(f4,2)>0
                                  x=XYZ(f4,1)-DX; y=XYZ(f4,2); numc=numc+1;
                                  plot(x,y,'ko',xe,ye,'r.')
                             else
                                 plot(xe,ye,'b.'); nump=nump+1;
                             end
                             xlabel('X'); ylabel('Y');
                             title('UD muons')
                             axis([0 768 0 1536]); 
                             hold on
                         end
                    end  % if success
                    end % sumhits and minhits condition
                end % second hds in pair
            end % first hds in pair
          end
        end

   end
   if mod(i,1000)==0 || i==NS
fprintf('Episodes:%d; Muons:%d; UD:%d; UL:%d; UR:%d; LD:%d; RD:%d, LR:%d\n', ... 
            i,NE,Iudlr(1:6)); 
if DISP(7)==1
fprintf('Points bad: %d, good: %d , total: %d\n', nump, numc, nump+numc);
end
  end
  
end

EVENTS=EVENTS(1:we,:);
% disp(Stat);
% Statistical report:
fprintf('Number of HDS in all events = %d, \n',  NumHDS);
f1=find(NumQuadsPerHDS>0,1,"last"); 
NumQuadsPerHDS=NumQuadsPerHDS(1:f1);
f2=find(NumHitsPerHDScloud>0,1,'last');
NumHitsPerHDScloud=NumHitsPerHDScloud(1:f2);
f3=find(NumHitsByMuons>0,1,'last');
NumHitsByMuons=NumHitsByMuons(1:f3);

fprintf('Average number of quads per HDS= %f\n', ...
                                sum(NumQuadsPerHDS.*(1:f1))/NumHDS);
fprintf('The same after list quantization=%f\n', ... 
                                sum(NumHitsPerHDScloud.*(1:f2))/NumHDS); 
fprintf('Average number of hits by detected muons=%f\n',...
                                sum(NumHitsByMuons.*(1:f3))/NumHDS);     
H=768;
TotalPoints=3*3*H*2;  %  3 panels, 2H for Pikk and 4H for Lai
MaxGoodHits=6*NumHDS;
DetectedHits=sum(NumHitsByMuons.*(1:f3));
TotalHits=sum(NumHitsPerHDScloud.*(1:f2));
ProbFalseHit=(TotalHits-DetectedHits)/TotalPoints/NumHDS;
ProbLostHit=1-DetectedHits/MaxGoodHits;

fprintf('False hit probability per fiber = %e \n',ProbFalseHit);
fprintf('Lost hit probability per fiber = %e \n',ProbLostHit);

if DISP(8)>0
    figure(4); clf(4); 
    subplot(1,3,1);  bar(1:100,NumQuadsPerHDS(1:100));
    title('number of quads \newline per HDS')
    subplot(1,3,2);  bar(1:200,NumHitsPerHDScloud(1:200));
    title('number of hits per HDS \newline after list quantization')
    subplot(1,3,3);  bar(1:6,NumHitsByMuons);
    title('number of detected hits per \newline event after list quantization')
end



XYstat=XYstat(1:Jxy);
save XYstat XYstat;



return
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
function [XY,F,En]=transform(xy,mode,T)
 % Sum-matching
s12=xy(1)+xy(2); s34=xy(3)+xy(4);
d=s12-s34;        % matching  
dz=min(s12,s34);  % avoid division by zero
F=(abs(d)<T & dz>1e-6); % matched and non-zero   
if F
    XY=zeros(1,2,'single');  % transform
    switch mode
        case 1                 % GSCAN
            XY(1)=(xy(1)-xy(2))./s12;
            XY(2)=(xy(3)-xy(4))./s34;
            XY=4*(XY+1); % [-1 1]->[0 2]-> [0 8]
        case 2                 % suggested
            XY(1)=xy(1)./s12;
            XY(2)=xy(3)./s34;
            XY=XY*8;     % [0 1]->[0 8]
    end
else
    XY=[0 0 0 0]; % F is empty
end
En=(s12+s34)/2;
%--------------------------------------------------------------------------
function QTABLE=qtable(scale)
m=length(scale);
for i=1:m-1
    QTABLE(scale(i)+1:scale(i+1))=i-1;
end

%--------------------------------------------------------------------------
function [FibList, FibMetr, Jxy] = ... 
                      fibCloud(mpi,X,Y,IS,IE,MaxErr,QTABLE,E)
% List quantization
% Inputs:
%    mpi are combined muxplate&mux indices
%    X,Y are transformed measurement pairs
%    IS, IE are start/finish pairs of time stamps of input quadruples 
%    QTABLE is quantization table
%    E are energies of measurements
% Outputs:
%    FibList   is list of fibers
%    FibMetr   their metrics
%    Jxy       start/finish indices

m=sum(mpi>=0); % actual number of input samples
j=0;           % counter of output samples
% Prepare arrays
FibList=-ones(1,9*m); FibMetr=-ones(1,9*m); Jxy=-ones(2,9*m);  
% Loop over samples
for i=1:m
    x=QTABLE(round(X(i)*10)+1); % table-based quantization
    switch x
        case 0, XX=0:1;                      % list   
        case {1,2,3,4,5,6},  XX=[x-1 x x+1]; % list
        case {7,8,9}, XX=6:7;                % list  
    end
    % XX=QTABLE(round(X(i)*10)+1,:);
    y=QTABLE(round(Y(i)*10)+1);
    switch y
        case 0, YY=0:1;                      % list
        case {1,2,3,4,5,6}, YY= [y-1 y y+1]; % list
        case {7,8,9}, YY=6:7;                % list 
    end
    % YY=QTABLE(round(Y(i)*10)+1,:);
    for ix=1:length(XX)      % X-list  
        for iy=1:length(YY)  % Y-list
            j=j+1;
            FibList(j)=mpi(i)+XX(ix)*4+YY(iy)*32;  % fiber numbers
            FibMetr(j)=((XX(ix)-X(i))^2+(YY(iy)-Y(i))^2)/E(i)^2; % error
            Jxy(:,j)=[IS(i); IE(i)];        % time interval
        end
    end
end
% restriction by error
f=FibMetr<MaxErr;        % positions with required error
FibList=FibList(f);      % reduce list
FibMetr=FibMetr(f);% I); % metrics  
Jxy=Jxy(:,f);            % time indices
% sort the list by fiber numbers
[FibList,I]=sort(FibList(FibList>=0));
FibMetr=FibMetr(I);
Jxy=Jxy(:,I);
return
%--------------------------------------------------------------------------
function Coord = fib2gcs(FiberIndex,hds,mat)
% 
% Fiber numbers bounds
%            P1  L1      P2  L2      P3   L3      
ZeroFiber=[   1 1537   4609 6145    9217 10753];
LastFiber=[1536 4608   6144 9216   10752 13824];
if mod(mat,2)==0 % Lai
    Coord=(FiberIndex-ZeroFiber(mat))/2;  % lai
else             % Pikk
    switch hds
        case {1,4}
            Coord=(LastFiber(mat)-FiberIndex)/2;  % pikk in mm
        case {2,3}
            Coord=(FiberIndex-ZeroFiber(mat))/2;  % pikk in mm
    end
end
return
%--------------------------------------------------------------------------
function [ListCoord,ListMetr,JS,JF]=...
    GetCloud(MPI,X,Y,IS,IE,hds,EQmax,QTABLE,E)
% Input:
%   MPI, X,Y are matrices of size 6 x # of candidates
%   IS, IE   are start and end indices 
%   hds      is hodoscope number
%   EQmax=2; is maximum quantization error
%   QTABLE   is quantization table 
%   E        is Energy, not used 
% Output:
%   ListCoord are global coordinates
%   ListMetr 
%   JS,JF     are start and finish indices for list

nmax=max(sum(MPI>=0,2));    % max number of cand in mats
ListMetr=-ones(6,9*nmax);   % arrays for metrics
ListCoord=-ones(6,9*nmax);  %        and coordinates
JS=-ones(6,9*nmax);    % arrays for starts
JF=-ones(6,9*nmax);    % and finish 
for j=0:1  % pikk, lai loop
    % Get line through 3 planes in one dimension, Pikk or Lai 
    jj=[1 3 5]+j;    % mut numbers  
    nlist=[0 0 0];   % array for list sizes
    for i=1:3        % loop over mats 
        ij=jj(i);    % mat number
        % List of fibers
        [list, metr,Jxy] = ... % global coordinates 
            fibCloud(MPI(ij,:),X(ij,:),Y(ij,:),IS(ij,:),IE(ij,:),EQmax,QTABLE,E);
        nlist(i)=length(list);         % list lengths
        ListMetr(ij,1:nlist(i))=metr;  % metrics of list members
        fiblist=fib2gcs(list,hds,ij);  % Coordinates in mm
        % sort
        [fiblist,J]=sort(fiblist);
        ListCoord(ij,1:nlist(i))=fiblist;
        JS(ij,1:nlist(i))=Jxy(1,J);    % start 
        JF(ij,1:nlist(i))=Jxy(2,J);    % finish
    end
end