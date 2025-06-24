import React, { useEffect, useRef, useState } from 'react';
import {
  View,
  Button,
  Text,
  TextInput,
  StyleSheet
} from 'react-native';
import {
  RTCPeerConnection,
  RTCView,
  mediaDevices,
  MediaStream,
} from 'react-native-webrtc';
import { startMediasoup, startStreaming } from './mediasoupClient';


export default function App() {
  const [joined, setJoined] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [roomCode, setRoomCode] = useState('a');
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [remoteStreams, setRemoteStreams] = useState<MediaStream[]>([]);

  const joinMeeting = async () => {
    try {
      const mediaStream = await mediaDevices.getUserMedia({
        video: true,
        audio: true
      });
      setStream(mediaStream);

      await startMediasoup(roomCode, (remoteStream, kind) => {
        setRemoteStreams(prev => [...prev, remoteStream]);
      });

      await startStreaming(mediaStream, roomCode);
      setJoined(true);
    } catch (err: any) {
      console.error('Join failed:', err);
      setError(err.message);
    }
  };

  return (
    <View style={styles.container}>
      {!joined ? (
        <>
          <TextInput
            style={styles.input}
            placeholder="Enter meeting code"
            value={roomCode}
            onChangeText={setRoomCode}
          />
          <Button title="Join" onPress={joinMeeting} />
        </>
      ) : (
        <>
          <Text style={styles.status}>
            You're in the meeting
          </Text>
          {stream && (
            <RTCView
              streamURL={stream.toURL()}
              style={styles.video}
              objectFit="cover"
            />
          )}
          {remoteStreams.map((remoteStream, idx) => (
            <RTCView
              key={`remote-${idx}`}
              streamURL={remoteStream.toURL()}
              style={styles.video}
              objectFit="cover"
            />
          ))}
        </>
      )}
      {error && <Text style={styles.error}>{error}</Text>}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, padding: 16, justifyContent: 'center' },
  input: {
    borderWidth: 1, borderColor: '#ccc',
    padding: 8, marginBottom: 12, borderRadius: 6
  },
  status: { fontSize: 18, marginBottom: 12, textAlign: 'center' },
  video: { width: '100%', height: 200, backgroundColor: '#000', marginBottom: 12 },
  error: { color: 'red', textAlign: 'center', marginTop: 10 }
});
