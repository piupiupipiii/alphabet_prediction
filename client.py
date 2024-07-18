import socket


def client_program():
    host = socket.gethostname()
    port = 5000

    client_socket = socket.socket()
    client_socket.connect((host, port))

    print(f"Success connect to {host}:{port}")

    while True:
        message = input(" -> ")

        if message.lower().strip() == 'bye':
            print("Shutting down...")
            break

        client_socket.send(message.encode())
        print("Sending message...")
        print("Waiting message from server...")

        data = client_socket.recv(1024).decode()
        print('Message from server: ' + data)

    client_socket.close()


if __name__ == '__main__':
    client_program()
