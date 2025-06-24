import React, { useEffect, useRef, useState } from 'react';
import {
  View,
  Button,
  Text,
  TextInput,
  StyleSheet,
  Platform
} from 'react-native';
import {
  RTCPeerConnection,
  RTCView,
  mediaDevices,
  MediaStream,
} from 'react-native-webrtc';
import {
  request,
  PERMISSIONS,
  RESULTS,
} from 'react-native-permissions';
import { startMediasoup, startStreaming } from './mediasoupClient';



export default function App() {
  const [joined, setJoined] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [roomCode, setRoomCode] = useState('a');
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [remoteStreams, setRemoteStreams] = useState<MediaStream[]>([]);

  useEffect(() => {
    requestPermissions();
    testServerConnectivity();
  }, []);

  const testServerConnectivity = async () => {
    const servers = [
      { name: 'Server 1 (Port 8000)', url: 'http://10.0.2.2:8000' },
      { name: 'Server 2 (Port 3000)', url: 'http://10.0.2.2:3000' },
    ];

    for (const server of servers) {
      try {
        const response = await fetch(server.url);
        if (response.ok) {
          console.log(`${server.name} is accessible`);
        } else {
          console.log(`${server.name} returned an error`);
        }
      } catch (error) {
        console.error(`${server.name} is not accessible`, error);
      }
    }
  };

  const requestPermissions = async () => {
    try {
      // Request Camera Permission
      const cameraStatus = await request(
        Platform.OS === 'ios'
          ? PERMISSIONS.IOS.CAMERA
          : PERMISSIONS.ANDROID.CAMERA
      );

      if (cameraStatus === RESULTS.GRANTED) {
        console.log('Camera permission granted');
      } else {
        console.log('Camera permission denied');
      }

      // Request Microphone Permission
      const microphoneStatus = await request(
        Platform.OS === 'ios'
          ? PERMISSIONS.IOS.MICROPHONE
          : PERMISSIONS.ANDROID.RECORD_AUDIO
      );

      if (microphoneStatus === RESULTS.GRANTED) {
        console.log('Microphone permission granted');
      } else {
        console.log('Microphone permission denied');
      }
    } catch (error) {
      console.error('Error requesting permissions:', error);
    }
  };

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
      setError(err.message || 'An unknown error occurred');
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
