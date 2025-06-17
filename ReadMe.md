##Backend:
cd backend
open constants.ts file, replace announced ip with your ipv4 address
run command: npm run start:dev


##Frontend:
cd frontend
run command: npm run dev

##python:
cd ml
run command: python webrtc_python.py


instructions:
run backend before you run python server
** Use this to get ipv4 address ' ip -4 addr show '
paste this in constants.ts (backend/src/calls/constants.ts) file, at "announcedIp" key