% Type config
function TypeConfig(Tq,TS,EQmax,Tr_mode,Td)
if nargin<8
fprintf('Parameters of the algorithm: \n');
fprintf('Threshold for matching X+,X- with Y+,Y- = %3.2f \n',Tq);
fprintf('Time window for quads                   = %3.2f ns\n',TS/1000);
fprintf('Quantization error threshold            = %3.2f \n', EQmax);
fprintf('Transform 1/2: Original/Alternative     = %d \n',Tr_mode);
fprintf('Maximum deviations                      = %3.2f,%3.2f\n',...
                                    Td(1), Td(2));
else
fid=fopen(filename,'w');
fprintf(fid,'Parameters of the algorithm: \n');
fprintf(fid,'Threshold for matching X+,X- with Y+,Y- = %3.2f \n',Tq);
fprintf(fid,'Time window for quads                   = %3.2f ns\n',TS/1000);
fprintf(fid,'Quantization error threshold            = %3.2f \n',MaxErr);
fprintf(fid,'Transform 1/2: Original/Alternative     = %d \n',Tr_mode);
fprintf(fid,'Maximum deviations                      = %3.2f,%3.2f\n',...
                                    Td(1), Td(2));
fclose(fid);
end

 