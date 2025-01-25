from flask import Flask, render_template, request, jsonify
import json
from scipy.spatial.distance import cosine

app = Flask(__name__)

# Load character profiles
with open("char_scores/got_char_profiles.json", "r") as f:
    character_profiles = json.load(f)

# Define scoring logic (your existing code here)
# Define quiz questions and answer options with scores
questions = [
    # loyalty
    "Would you break a promise to protect a loved one, even if it had serious consequences?",
    "How important is it to be trustworthy to you?",
    "Would you betray a friend to save yourself from trouble?",
    "How do you feel about people who break their oaths?",
    "Would you sacrifice your own happiness for the happiness of your family?", 
    # ambition
    "What is your primary motivation in life?",
    "How important is it to you to be successful?",
    "Would you be willing to take risks to achieve your goals?",
    "How do you feel about competition?",
    "Do you believe that everyone deserves an equal opportunity to succeed?",
    # manipulativeness
    "Do you often try to influence other people's opinions?",
    "How comfortable are you with using flattery to get what you want?",
    "Do you often try to anticipate and exploit other people's weaknesses?",
    "How important is it to you to maintain a good public image?",
    "Would you ever lie to protect yourself from harm?",
    # courage
    "How do you react to danger or threats?",
    "Would you stand up for someone being bullied?",
    "How do you feel about taking risks?",
    "Do you admire people who overcome adversity?",
    "How would you describe your own level of bravery?"
]

answer_options = {
    questions[0]: ["Absolutely not.","It depends on the situation.", "If it benefits me in the long run, maybe.", "Promises are easily broken."],
    questions[1]: ["Extremely important. My word is my bond.", "Important, but not the most important thing.", "Not very important. People should look out for themselves.", "Not very important. People should look out for themselves.", "Trust is overrated."],
    questions[2]: ["Never", "Maybe, if the consequences were severe enough.", "Definitely, if it meant my survival.", "I don't have many close friends to betray."],
    questions[3]: ["I have no respect for them.", "It depends on the reason for breaking the oath.", "It's understandable if they had no choice.", "Oaths are meaningless in the real world."],
    questions[4]: ["Absolutely, without hesitation.", "I would consider it, but it would be a difficult decision.", "My own happiness is important to me.", "Family is important, but not more important than myself."],
    questions[5]: ["To help others and make the world a better place.", "To achieve personal and professional success.", "To live a comfortable and peaceful life.", "To experience as much as possible and have fun."],
    questions[6]: ["Very important. I strive to be the best at everything I do.", "Important, but not the most important thing in life.", "Not very important. I'm content with a simple life.", "Success is subjective and not something I focus on."],
    questions[7]: ["Yes, I'm not afraid to take calculated risks.", "I prefer to play it safe and avoid unnecessary risks.", "I would only take risks if the potential rewards are very high.", "I'm not very ambitious and rarely take risks."],
    questions[8]: ["I thrive on competition and always strive to win.", "I'm comfortable with competition, but it's not my primary focus.", "I prefer to avoid competition whenever possible.", "I don't understand the appeal of competition."],
    questions[9]: ["Absolutely. Everyone should have a fair chance.", "Yes, but some people are naturally more talented than others.", "Not necessarily. Some people are simply not meant to succeed.", "I don't really have strong opinions on this."],
    questions[10]: ["No, I value honesty and direct communication.", "Sometimes, depending on the situation.", "Yes, I enjoy persuading others to see things my way.", "I'm not good at manipulating people and don't usually try."],
    questions[11]: ["Very uncomfortable. I find it dishonest.", "Somewhat uncomfortable, but I might do it in certain situations.", "I don't mind using flattery if it gets me the results I want.", "I'm not above using flattery to achieve my goals."],
    questions[12]: ["No, I believe in treating others with respect.", "Sometimes, but only if it's necessary.", "Yes, I think it's a valuable skill to be able to read people.", "I'm not very good at reading people and don't usually try to exploit them."],
    questions[13]: ["Not very important. I am who I am.", "Important, but I wouldn't do anything dishonest to protect it.", "Very important. My reputation is crucial to my success.", "I don't really care what other people think of me."],
    questions[14]: ["No, I would always tell the truth.", "It depends on the situation.", "Yes, if it meant avoiding serious consequences.", "I'm not afraid to lie if it benefits me."],
    questions[15]: ["I face my fears head-on.", "I try to stay calm and assess the situation.", "I tend to avoid dangerous situations if possible.", "I easily panic and get overwhelmed by fear."],
    questions[16]: ["Absolutely. I wouldn't tolerate injustice.", "I would try to intervene if it was safe to do so.", "I would probably try to stay out of it.", "I would be too afraid to get involved."],
    questions[17]: ["I embrace challenges and enjoy taking calculated risks.", "I'm willing to take risks if the potential rewards outweigh the dangers.", "I prefer to avoid unnecessary risks.", "I'm generally risk-averse and prefer safety."],
    questions[18]: ["Yes, I find their resilience inspiring.", "I admire their strength, but I wouldn't want to go through what they did.", "I don't necessarily admire them. Everyone faces challenges.", "I don't pay much attention to other people's struggles."],
    questions[19]: ["I consider myself to be very courageous.", "I'm reasonably courageous, but I have my limits.", "I'm not a particularly courageous person.", "I would describe myself as a coward."]
}
answer_scores = {
    "loyalty": {
        questions[0]: {"Absolutely not.": 1, "It depends on the situation.": 0.75, "If it benefits me in the long run, maybe.": 0.25, "Promises are easily broken.": 0},
        questions[1]: {"Extremely important. My word is my bond.": 1, "Important, but not the most important thing.": 0.75, "Not very important. People should look out for themselves.": 0.25, "Trust is overrated.": 0},
        questions[2]: {"Never": 1, "Maybe, if the consequences were severe enough.": 0.5, "Definitely, if it meant my survival.": 0, "I don't have many close friends to betray.": 0},
        questions[3]: {"I have no respect for them.": 1, "It depends on the reason for breaking the oath.": 0.75, "It's understandable if they had no choice.": 0.5, "Oaths are meaningless in the real world.": 0},
        questions[4]: {"Absolutely, without hesitation.": 1, "I would consider it, but it would be a difficult decision.": 0.75, "My own happiness is important to me.": 0.5, "Family is important, but not more important than myself.": 0.25}
    },
    "ambition": {
        questions[5]: {"To help others and make the world a better place.": 0.25, "To achieve personal and professional success.": 1, "To live a comfortable and peaceful life.": 0.5, "To experience as much as possible and have fun.": 0.25},
        questions[6]: {"Very important. I strive to be the best at everything I do.": 1, "Important, but not the most important thing in life.": 0.75, "Not very important. I'm content with a simple life.": 0.25, "Success is subjective and not something I focus on.": 0},
        questions[7]: {"Yes, I'm not afraid to take calculated risks.": 1, "I prefer to play it safe and avoid unnecessary risks.": 0.5, "I would only take risks if the potential rewards are very high.": 0.25, "I'm not very ambitious and rarely take risks.": 0},
        questions[8]: {"I thrive on competition and always strive to win.": 1, "I'm comfortable with competition, but it's not my primary focus.": 0.75, "I prefer to avoid competition whenever possible.": 0.25, "I don't understand the appeal of competition.": 0},
        questions[9]: {"Absolutely. Everyone should have a fair chance.": 1, "Yes, but some people are naturally more talented than others.": 0.75, "Not necessarily. Some people are simply not meant to succeed.": 0.25, "I don't really have strong opinions on this.": 0}
    },
    "manipulativeness": {
        questions[10]: {"No, I value honesty and direct communication.": 0, "Sometimes, depending on the situation.": 0.5, "Yes, I enjoy persuading others to see things my way.": 1, "I'm not good at manipulating people and don't usually try.": 0.25},
        questions[11]: {"Very uncomfortable. I find it dishonest.": 0, "Somewhat uncomfortable, but I might do it in certain situations.": 0.25, "I don't mind using flattery if it gets me the results I want.": 0.75, "I'm not above using flattery to achieve my goals.": 1},
        questions[12]: {"No, I believe in treating others with respect.": 0, "Sometimes, but only if it's necessary.": 0.25, "Yes, I think it's a valuable skill to be able to read people.": 1, "I'm not very good at reading people and don't usually try to exploit them.": 0.25},
        questions[13]: {"Not very important. I am who I am.": 0, "Important, but I wouldn't do anything dishonest to protect it.": 0.5, "Very important. My reputation is crucial to my success.": 1, "I don't really care what other people think of me.": 0.25},
        questions[14]: {"No, I would always tell the truth.": 0, "It depends on the situation.": 0.5, "Yes, if it meant avoiding serious consequences.": 1, "I'm not afraid to lie if it benefits me.": 1},
        questions[15]: {"I face my fears head-on.": 0, "I try to stay calm and assess the situation.": 0.5, "I tend to avoid dangerous situations if possible.": 0.25, "I easily panic and get overwhelmed by fear.": 0}
    },
    "courage": {
        questions[16]: {"Absolutely. I wouldn't tolerate injustice.": 1, "I would try to intervene if it was safe to do so.": 0.75, "I would probably try to stay out of it.": 0.25, "I would be too afraid to get involved.": 0},
        questions[17]: {"I embrace challenges and enjoy taking calculated risks.": 1, "I'm willing to take risks if the potential rewards outweigh the dangers.": 0.75, "I prefer to avoid unnecessary risks.": 0.25, "I'm generally risk-averse and prefer safety.": 0},
        questions[18]: {"Yes, I find their resilience inspiring.": 1, "I admire their strength, but I wouldn't want to go through what they did.": 0.75, "I don't necessarily admire them. Everyone faces challenges.": 0.25, "I don't pay much attention to other people's struggles.": 0},
        questions[19]: {"I consider myself to be very courageous.": 1, "I'm reasonably courageous, but I have my limits.": 0.75, "I'm not a particularly courageous person.": 0.25, "I would describe myself as a coward.": 0}
    }
}

def get_user_answers():
    """Gets user answers to the quiz questions."""
    user_answers = {}
    for i, question in enumerate(questions):
        print(f"{i+1}. {question}")
        for j in range(4):
            f"{question}"
            print(f"{j+1}. {answer_options[question][j]}")
        user_choice = input("Select your answer (1, 2, 3, or 4): ").lower()
        while user_choice not in ["1", "2", "3", "4"]:
            print("Invalid input. Please enter '1', '2', '3', or '4'.")
            user_choice = input("Select your answer (1, 2, 3, or 4): ").lower()
        user_answers[question] = int(user_choice)-1
        print()
    return user_answers

def calculate_user_profile(user_answers):
    """Calculates user's personality profile."""
    user_profile = {"loyalty": 0, "ambition": 0, "manipulativeness": 0, "courage": 0}
    for question, answer in user_answers.items():
        for trait, trait_scores in answer_scores.items():
            if question in trait_scores:
                user_profile[trait] += trait_scores[question][answer_options[question][answer]]
    return user_profile

def find_closest_character(user_profile, character_profiles):
    """Finds the closest character based on cosine similarity."""
    min_distance = float('inf')
    closest_character = None

    for character, character_profile in character_profiles.items():
        #distance = cosine(list(user_profile.values()/4), list(character_profile.values()))
        distance = cosine(list(value / 4 for value in user_profile.values()), list(character_profile.values()))
        if distance < min_distance:
            min_distance = distance
            closest_character = character
            print("character: ", character)
            print("character distance: ", distance)
    print("user_profile: ", user_profile)

    return closest_character, min_distance

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/submit", methods=["POST"])
def submit():
    user_answers = request.json.get("answers", {})
    result = calculate_user_profile(user_answers)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
