from qiskit.circuit import QuantumCircuit, QuantumRegister, AncillaRegister,ClassicalRegister, Parameter, ParameterVector

def simple_circ(qc:QuantumCircuit):
    qubits = qc.qubits

    n = qc.num_qubits
    para = ParameterVector('θ', 2*n).params

    for i in range(n):
        qc.ry(para[i],qubits[i])
    qc.barrier()
    for i in range(n//2):
        qc.cnot(qubits[i*2],qubits[i*2+1])
    for i in range((n-1)//2):
        qc.cnot(qubits[i*2+1],qubits[i*2+2])
    qc.barrier()
    for i in range(n):
        qc.ry(para[i+n],qubits[i])

    return qc, para

class VCircuitConstructor:
    def __init__(self, n:int, ansatz:str='simple'):
        self.n = n

        if ansatz == 'simple':
            self.ansatz = simple_circ
        elif isinstance(ansatz, function):
            self.ansatz = ansatz
        else:
            raise ValueError('Invalid ansatz.')
    
    def get_circuit(self):
        qubits = QuantumRegister(self.n)
        qc,para = self.ansatz(QuantumCircuit(qubits))

        return {
            'circuit': qc,
            'qubits': qubits,
            'para': para
        }
    
    @staticmethod
    def get_vqc(n:int, ansatz:str='simple'):
        qubits = QuantumRegister(n)
        if ansatz == 'simple':
            ansatz = simple_circ

        qc,para = ansatz(QuantumCircuit(qubits))

        return {
            'circuit': qc,
            'qubits': qubits,
            'para': para
        }
