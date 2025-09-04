from scapy.all import sniff
import scapy 
import threading
import socket

# Obtém o IP local
# capturar os pacotes e possivel mais teria que identificar dados do dataset apartir deles
# oque e bem complicado
local_ip = socket.gethostbyname(socket.gethostname())


def show_packet(packet):

    
    port_info = ""
    
    # Verifica se o pacote é TCP
    if packet.haslayer("TCP"):
        sport = packet["TCP"].sport
        dport = packet["TCP"].dport
        port_info = f"Portas: {sport} -> {dport}"
    
    # Verifica se o pacote é UDP
    elif packet.haslayer("UDP"):
        sport = packet["UDP"].sport
        dport = packet["UDP"].dport 
        port_info = f"Portas: {sport} -> {dport}"
    
    # Exibe o resumo do pacote e as informações das portas
    # print(packet.summary() + "\n PORTA" + port_info + "\n" + packet.show(dump=True) + "\n" + "-" * 80)
    print(packet.show())

def start_sniffing():
    """
    Captura pacotes de rede em tempo real e exibe todos os dados dos pacotes recebidos da internet.
    A execução fica em loop até interrompida manualmente (CTRL+C).
    """
    print("Iniciando a captura de pacotes destinados ao IP:", local_ip)
    print("Pressione CTRL+C para interromper.")
    # Filtro para capturar apenas pacotes destinados ao IP local
    # sniff(prn=process_packet, store=False, filter=f"dst host {local_ip} or dst host {local_ip}/24")
    # Sem filtro (so o firewall do SO)
    sniff(prn=process_packet, store=False)
def process_packet(packet):
    # Cria uma nova thread para mostrar o pacote
    thread = threading.Thread(target=show_packet, args=(packet,))
    thread.start()

if __name__ == "__main__":
    start_sniffing()
