// udp-sender.js
const dgram = require("dgram");
const client = dgram.createSocket("udp4");

const message = Buffer.from("hello from sender");
client.send(message, 25000, "127.0.0.1", (err) => {
  if (err) console.error(err);
  else console.log("✅ Sent test UDP message to 25000");
  client.close();
});
