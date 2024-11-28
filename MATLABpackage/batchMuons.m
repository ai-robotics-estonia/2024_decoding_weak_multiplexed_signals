clear
addpath data; addpath lib; % clf;
% Wiki:About 10,000 muons reach every square meter of the 
% earth's surface a minute

% Parameters 
TS=50000;    % event search window
Tq=2;        % Quadruple matching precision |dx-dy|<Tq
EQmax=2;     % Maximum allowed quantization error per point 
Tr_mode=2;   % 1=conventional,2=alternative transform
Td=2*[1 1];  % (32,8);  % deviations between true and extrapolation
%  Td(1) for UD and LR, Td(2) for all others UL,UR,DL,DR
%  Source data: short/long/middle
mode='short'; %   Data file name
% mode='long';  %   Data file name
% mode='middle';  %   Data file name
TypeConfig(Tq,TS,EQmax,Tr_mode,Td);
DISP=[0 0 0 0 0 0 0 0];   % 1/0 Show/hide debug info and plots 
% DISP=[1 1 1 1 1 1 0];   % 1/0 Show/hide debug info and plots 
Trajectory=[1 1 1 1 1 1]; % which directions are to be processed
TypeEvents=1;          % Create output file

OutFileVersion=2;
% Version 1: both Points and Approximations are printed with masks
% Version 2: One version is printed. Non-available points are 
%             filled by approximated versions

% Read input data file  
switch mode
    case 'short',  DirName='tmp_short'; fname='data\Short'; % load Short.mat;
    case 'middle', DirName='tmp_middle';fname='data\Middle'; % load Middle.mat; 
    case 'long',   DirName='tmp_long';  fname='data\Long';  % load Long.mat;
end
if ~exist(DirName,'dir'), mkdir(DirName); end
%---------------
% Main routine
%---------------
EVENTS = stream2events(fname,TS,Tq,EQmax,Tr_mode,Td,... 
                       DISP, Trajectory);
if TypeEvents==0, return; end
%------------------------------
% Print list of events
%------------------------------
fname=[DirName '\EVENTS.txt']; save(fname,"EVENTS");
fid=fopen(fname,"w");

switch OutFileVersion
    case 1
fprintf(fid,"    #   X(mm)   Y(mm)   Z(mm)"  );
fprintf(fid,"   XA      YA      ZA     Mask   Time(ns) ID \n");
for i=1:size(EVENTS,1) 
    % fprintf("%4d %8.2f  %8.2f  %8.2f  %8.2f        %2d \n",...
    %         int16(EVENTS(i,1)),EVENTS(i,2:5), int16(EVENTS(i,6))); 
    fprintf(fid,"%5d %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f %2d%2d%2d %6.2f  %2d \n",...
            int16(EVENTS(i,1)),EVENTS(i,2:4),EVENTS(i,5:7),...
            EVENTS(i,8:10), EVENTS(i,11),EVENTS(i,12));    
end
fclose(fid);
    case 2
fprintf(fid,"    #   X(mm)   Y(mm)   Z(mm)  Mask  Time(ns) ID \n");
for i=1:size(EVENTS,1)
    P=EVENTS(i,2:4); PA=EVENTS(i,5:7); mask=EVENTS(i,8:10);
    P(mask==0)=PA(mask==0);
    fprintf(fid,"%5d %7.2f %7.2f %7.2f  %2d%2d%2d %6.2f  %2d \n",...
            int16(EVENTS(i,1)),P,mask, EVENTS(i,11:12));    
end
end % switch 
        