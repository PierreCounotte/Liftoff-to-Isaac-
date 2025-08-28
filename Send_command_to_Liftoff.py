import socket
import struct
import time

# Paramètres réseau
UDP_IP = "127.0.0.1"   # Adresse locale
UDP_PORT = 5000        # Port défini dans Liftoff (à configurer dans les paramètres)

# Crée un socket UDP
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def send_command(throttle, yaw, pitch, roll):
    """
    Envoie une commande normalisée (-1 à 1) au simulateur Liftoff
    """
    # Paquet en float32, format (throttle, yaw, pitch, roll)
    data = struct.pack('ffff', throttle, yaw, pitch, roll)
    sock.sendto(data, (UDP_IP, UDP_PORT))

print("Démarrage du contrôle du drone via UDP...")

try:
    for i in range(300):  # Envoie des commandes pendant 5 secondes (60 Hz)
        throttle = 0.5    # Montée lente
        yaw = 0.0
        pitch = 0.0
        roll = 0.0
        send_command(throttle, yaw, pitch, roll)
        time.sleep(1/60)
except KeyboardInterrupt:
    print("Arrêt manuel")
finally:
    sock.close()
    print("Connexion fermée")
