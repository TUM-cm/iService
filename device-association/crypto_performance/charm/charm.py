'''
Installation

dev for python 3
2.7-dev for python 2.7

python3
python3-dev
python3-setuptools

apt-get install libgmp-dev
apt-get install build-essential flex bison

wget https://crypto.stanford.edu/pbc/files/pbc-0.5.14.tar.gz
tar xf pbc-0.5.14.tar.gz
cd pbc
./configure
make
sudo make install

apt-get install openssl
 
cd charm
./configure.sh
make install
make test

Test:
python3 charm.py

Paillier and El Gamal 
    homomorphic: +, *
    non-homomorphic: -, /
'''

def paillier_cryptosystem():
    from charm.toolbox.integergroup import RSAGroup
    from charm.schemes.pkenc.pkenc_paillier99 import Pai99

    group = RSAGroup()
    pai = Pai99(group)
    (public_key, secret_key) = pai.keygen()
    
    operand_1 = 3
    operand_2 = 4
    encode_msg_1 = pai.encode(public_key['n'], operand_1)
    encode_msg_2 = pai.encode(public_key['n'], operand_2)       
    cipher_1 = pai.encrypt(public_key, encode_msg_1)
    cipher_2 = pai.encrypt(public_key, encode_msg_2)
    
    # supported
    paillier_plus(pai, public_key, secret_key, operand_1, operand_2, cipher_1, cipher_2)
    paillier_mult(pai, public_key, secret_key, operand_1, operand_2, cipher_1, cipher_2)
    
    # unsupported
    paillier_minus(pai, public_key, secret_key, operand_1, operand_2, cipher_1, cipher_2)
    paillier_division(pai, public_key, secret_key, operand_1, operand_2, cipher_1, cipher_2)
    
def paillier_plus(pai, public_key, secret_key, operand_1, operand_2, cipher_1, cipher_2):
    try:
        print("########## + #############")
        unencrypted = operand_1 + operand_2
        decrypted = pai.decrypt(public_key, secret_key, cipher_1 + cipher_2)
        print(unencrypted)
        print(decrypted)
    except:
        pass

def paillier_mult(pai, public_key, secret_key, operand_1, operand_2, cipher_1, cipher_2):
    try:
        print("########## * #############")
        unencrypted = operand_1 * operand_2
        result = cipher_1 * cipher_2
        print(unencrypted)
        print(result)
        decrypted = pai.decrypt(public_key, secret_key, result)
        print(decrypted)
    except:
        pass

# Not supported
def paillier_minus(pai, public_key, secret_key, operand_1, operand_2, cipher_1, cipher_2):
    try:
        print("########## - #############")
        unencrypted = operand_1 - operand_2
        decrypted = pai.decrypt(public_key, secret_key, cipher_1 - cipher_2)
        print(unencrypted)
        print(decrypted)
    except:
        pass

def paillier_division(pai, public_key, secret_key, operand_1, operand_2, cipher_1, cipher_2):
    try:
        print("########## / #############")
        unencrypted = operand_1 / operand_2
        decrypted = pai.decrypt(public_key, secret_key, cipher_1 / cipher_2)
        print(unencrypted)
        print(decrypted)
    except:
        pass

def el_gamal_cryptosystem():
    from charm.toolbox.eccurve import prime192v2
    from charm.toolbox.ecgroup import ECGroup
    from charm.schemes.pkenc.pkenc_elgamal85 import ElGamal

    groupObj = ECGroup(prime192v2)
    el = ElGamal(groupObj)
    (public_key, secret_key) = el.keygen()
    msg = b"hello world!12345678"
    cipher_text = el.encrypt(public_key, msg)
    decrypted_msg = el.decrypt(public_key, secret_key, cipher_text)    
    print(decrypted_msg == msg)
    
def main():
    paillier_cryptosystem()
    el_gamal_cryptosystem()

if __name__ == "__main__":
    main()
