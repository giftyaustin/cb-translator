const dgram = require("dgram");
const server = dgram.createSocket("udp4");

const PORT = 25000;

server.on("message", (msg, rinfo) => {
  console.log(
    `📦 Got message from ${rinfo.address}:${rinfo.port} - Size: ${msg.length} bytes`
  );
});

server.bind(PORT, () => {
  console.log(`✅ Listening on UDP port ${PORT}`);
});

server.on("error", (err) => {
  console.error(`❌ UDP Server error:\n${err.stack}`);
  server.close();
});
