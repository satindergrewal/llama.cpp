#include "chat.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static std::string read_file(const std::string & path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to read " + path);
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

static common_chat_msg make_msg(const char * role, const char * content) {
    common_chat_msg m;
    m.role    = role;
    m.content = content;
    return m;
}

static std::string apply_prompt(const common_chat_templates * tmpls, std::vector<common_chat_msg> messages) {
    common_chat_templates_inputs inputs;
    inputs.messages              = std::move(messages);
    inputs.add_generation_prompt = true;
    return common_chat_templates_apply(tmpls, inputs).prompt;
}

static void require_before_first_user(const std::string & prompt, const char * prefix, const char * first_user) {
    const auto p = prompt.find(prefix);
    const auto u = prompt.find(first_user);
    if (p == std::string::npos) {
        throw std::runtime_error(std::string("missing prefix content: ") + prefix + "\n" + prompt);
    }
    if (u == std::string::npos) {
        throw std::runtime_error(std::string("missing user content: ") + first_user + "\n" + prompt);
    }
    if (p > u) {
        throw std::runtime_error(std::string(prefix) + " appeared after first user content\n" + prompt);
    }
}

int main() {
    auto tmpls = common_chat_templates_ptr(
        common_chat_templates_init(/* model= */ nullptr, read_file("models/templates/Qwen3.5-4B.jinja")));

    const auto user1 = make_msg("user", "USER_ONE");
    const auto user2 = make_msg("user", "USER_TWO");
    const auto sys   = make_msg("system", "SYS_PROMPT");
    const auto dev   = make_msg("developer", "DEV_PROMPT");

    {
        const auto prompt = apply_prompt(tmpls.get(), { user1, sys, user2 });
        require_before_first_user(prompt, "SYS_PROMPT", "USER_ONE");
        std::cout << "ok [user, system, user]\n";
    }
    {
        const auto prompt = apply_prompt(tmpls.get(), { user1, dev, user2 });
        require_before_first_user(prompt, "DEV_PROMPT", "USER_ONE");
        std::cout << "ok [user, developer, user]\n";
    }

    std::cout << "test-chat-system-hoist passed\n";
    return 0;
}
